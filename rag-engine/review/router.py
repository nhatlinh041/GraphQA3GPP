"""
HTTP surface of the human validation tool: /api/review/*.

Raters sign in with their registered name + their own password (set by an
admin, changeable by the rater) and get a signed 7-day token in X-Review-Token.
Admins are accounts in .env, REVIEW_ADMIN_USERS="user:password[,…]", and get a
12-hour token in X-Review-Admin; the username it carries is what adjudications
are attributed to. Token format and hashing: review/auth.py.

A wrong secret sleeps briefly before answering so nothing can be brute-forced
at request speed, and an unset REVIEW_SECRET makes every token invalid rather
than letting anything through.
"""
from __future__ import annotations

import os
import secrets
import time

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import BaseModel

from . import auth, store

router = APIRouter(prefix="/api/review", tags=["review"])


def _reject(status: int, msg: str) -> None:
    time.sleep(0.4)
    raise HTTPException(status, msg)


def _admin_users() -> dict[str, str]:
    out: dict[str, str] = {}
    for item in os.getenv("REVIEW_ADMIN_USERS", "").split(","):
        if ":" in item:
            u, pw = item.split(":", 1)
            if u.strip() and pw.strip():
                out[u.strip()] = pw.strip()
    return out


def rater(x_review_token: str | None = Header(default=None)) -> dict:
    parsed = auth.parse_token(x_review_token, "rater")
    r = store.get_reviewer(int(parsed[0])) if parsed and parsed[0].isdigit() else None
    if not parsed or not r or not r["active"] or not r["password_hash"] \
            or not auth.check_sig("rater", parsed[0], parsed[1], parsed[2], r["password_hash"]):
        _reject(401, "session expired or invalid — please sign in again")
    r.pop("password_hash", None)
    return r


def admin(x_review_admin: str | None = Header(default=None)) -> str:
    parsed = auth.parse_token(x_review_admin, "admin")
    pw = _admin_users().get(parsed[0]) if parsed else None
    if not parsed or not pw or not auth.check_sig("admin", parsed[0], parsed[1], parsed[2], pw):
        _reject(401, "admin session expired or invalid — please sign in again")
    return parsed[0]


# ---------------------------------------------------------------- rater

class LoginIn(BaseModel):
    name: str
    password: str


@router.post("/login")
def login(body: LoginIn) -> dict:
    r = store.get_reviewer_by_name(body.name)
    if not r or not r["active"] or not auth.verify_password(body.password, r["password_hash"]):
        _reject(401, "wrong name or password")
    token, exp = auth.make_token("rater", str(r["id"]), r["password_hash"], auth.RATER_TTL_S)
    store.log_event(r["name"], "login", None)
    return {"token": token, "expires_at": exp, "reviewer": {"id": r["id"], "name": r["name"], "role": r["role"]}}


class PasswordChangeIn(BaseModel):
    old_password: str
    new_password: str


@router.post("/me/password")
def change_password(body: PasswordChangeIn, r: dict = Depends(rater)) -> dict:
    full = store.get_reviewer(r["id"])
    if not auth.verify_password(body.old_password, full["password_hash"] if full else None):
        _reject(400, "current password is incorrect")
    if len(body.new_password) < 6:
        raise HTTPException(400, "new password must be at least 6 characters")
    store.set_password(r["id"], auth.hash_password(body.new_password))
    store.log_event(r["name"], "password_changed", None)
    # Old tokens die with the old hash; hand back a fresh one so the tab stays signed in.
    full = store.get_reviewer(r["id"])
    token, exp = auth.make_token("rater", str(r["id"]), full["password_hash"], auth.RATER_TTL_S)
    return {"ok": True, "token": token, "expires_at": exp}


@router.get("/me")
def me(r: dict = Depends(rater)) -> dict:
    return {"reviewer": {"id": r["id"], "name": r["name"], "role": r["role"]},
            "campaigns": store.campaigns_for_reviewer(r["id"])}


@router.get("/queue")
def queue(campaign: int = Query(...), r: dict = Depends(rater)) -> dict:
    items = store.queue_for(campaign, r["id"])
    with store.connect() as conn:
        lock = store._lock_state(conn, campaign, r["id"])
    return {"campaign": campaign, "items": items, "lock": lock,
            "n_done": sum(1 for i in items if i["status"] == "done"), "n_total": len(items)}


@router.get("/q/{qid}")
def question(qid: str, campaign: int = Query(...), r: dict = Depends(rater)) -> dict:
    try:
        return store.question_view(campaign, qid, r["id"])
    except store.Locked as e:
        raise HTTPException(423, str(e))
    except PermissionError:
        raise HTTPException(403, "this question is not assigned to you")
    except KeyError:
        raise HTTPException(404, "no such question")


@router.post("/q/{qid}/reveal")
def reveal(qid: str, campaign: int = Query(...), r: dict = Depends(rater)) -> dict:
    try:
        return store.reveal(campaign, qid, r["id"], r["name"])
    except store.Locked as e:
        raise HTTPException(423, str(e))
    except PermissionError:
        raise HTTPException(403, "this question is not assigned to you")
    except ValueError as e:
        raise HTTPException(400, str(e))


class RatingIn(BaseModel):
    status: str = "draft"
    c1: str | None = None
    c1_note: str | None = None
    c2: str | None = None
    c2_note: str | None = None
    irrelevant_chunks: list[str] | None = None
    own_answer: str | None = None
    c3: str | None = None
    verdict: str | None = None
    comment: str | None = None
    fix_text: str | None = None
    t_phase1_s: float | None = None
    t_phase2_s: float | None = None


@router.put("/q/{qid}/rating")
def put_rating(qid: str, body: RatingIn, campaign: int = Query(...), r: dict = Depends(rater)) -> dict:
    payload = body.model_dump(exclude_unset=True)
    status = payload.pop("status", "draft")
    try:
        out = store.upsert_rating(campaign, qid, r["id"], payload, status)
    except store.Locked as e:
        raise HTTPException(423, str(e))
    except PermissionError:
        raise HTTPException(403, "this question is not assigned to you")
    except ValueError as e:
        raise HTTPException(400, str(e))
    if status == "done":
        store.log_event(r["name"], "done", {"campaign": campaign, "qid": qid})
    return {"rating": out}


# ---------------------------------------------------------------- admin

class AdminLoginIn(BaseModel):
    username: str
    password: str


@router.post("/admin/login")
def admin_login(body: AdminLoginIn) -> dict:
    pw = _admin_users().get(body.username.strip())
    if not pw or not secrets.compare_digest(body.password.strip(), pw):
        _reject(401, "wrong username or password")
    token, exp = auth.make_token("admin", body.username.strip(), pw, auth.ADMIN_TTL_S)
    store.log_event(body.username.strip(), "admin_login", None)
    return {"token": token, "username": body.username.strip(), "expires_at": exp}


@router.get("/admin/me")
def admin_me(user: str = Depends(admin)) -> dict:
    return {"username": user}


class ReviewerIn(BaseModel):
    name: str
    role: str = "rater"
    password: str | None = None


class ReviewerPatch(BaseModel):
    name: str | None = None
    active: bool | None = None
    role: str | None = None


class PasswordIn(BaseModel):
    password: str | None = None


class CampaignIn(BaseModel):
    name: str
    sample_file: str
    raters_per_q: int = 2
    seed: int = 20261012
    reviewer_ids: list[int] = []
    all_raters: bool = False
    requires_campaign_id: int | None = None


class RequiresIn(BaseModel):
    requires_campaign_id: int | None = None


class AssignIn(BaseModel):
    reviewer_ids: list[int]


class AdjudicateIn(BaseModel):
    decision: str
    note: str | None = None
    decided_by: str | None = None


class StatusIn(BaseModel):
    status: str


@router.get("/admin/overview", dependencies=[Depends(admin)])
def admin_overview() -> dict:
    camps = store.list_campaigns()
    for c in camps:
        c["progress"] = store.progress(c["id"])
    return {"campaigns": camps, "reviewers": store.list_reviewers(), "samples": store.list_sample_files()}


@router.post("/admin/reviewers", dependencies=[Depends(admin)])
def add_reviewer(body: ReviewerIn) -> dict:
    """Creates the account. The password is the one given, or a generated one
    returned ONCE as `initial_password` — only the hash is stored."""
    password = (body.password or "").strip() or auth.generate_password()
    if len(password) < 6:
        raise HTTPException(400, "password must be at least 6 characters")
    try:
        r = store.add_reviewer(body.name, body.role, auth.hash_password(password))
    except ValueError as e:
        raise HTTPException(400, str(e))
    r["initial_password"] = password
    r["enrolled_campaigns"] = store.auto_enroll(r["id"]) if r["role"] == "rater" else []
    return r


@router.post("/admin/reviewers/{rid}/password", dependencies=[Depends(admin)])
def reset_password(rid: int, body: PasswordIn) -> dict:
    password = (body.password or "").strip() or auth.generate_password()
    if len(password) < 6:
        raise HTTPException(400, "password must be at least 6 characters")
    try:
        store.set_password(rid, auth.hash_password(password))
    except KeyError:
        raise HTTPException(404, "no such reviewer")
    return {"id": rid, "password": password}


@router.patch("/admin/reviewers/{rid}", dependencies=[Depends(admin)])
def patch_reviewer(rid: int, body: ReviewerPatch) -> dict:
    try:
        return store.update_reviewer(rid, body.name, body.active, body.role)
    except KeyError:
        raise HTTPException(404, "no such reviewer")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/admin/campaigns", dependencies=[Depends(admin)])
def create_campaign(body: CampaignIn) -> dict:
    try:
        return store.create_campaign(body.name, body.sample_file, body.raters_per_q, body.seed, body.reviewer_ids,
                                     all_raters=body.all_raters, requires_campaign_id=body.requires_campaign_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/admin/campaigns/{cid}/assign", dependencies=[Depends(admin)])
def assign(cid: int, body: AssignIn) -> dict:
    try:
        return {"per_reviewer": store.assign(cid, body.reviewer_ids), "progress": store.progress(cid)}
    except KeyError:
        raise HTTPException(404, "no such campaign")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.patch("/admin/campaigns/{cid}/status", dependencies=[Depends(admin)])
def set_status(cid: int, body: StatusIn) -> dict:
    if body.status not in ("open", "closed"):
        raise HTTPException(400, "status must be open|closed")
    if not store.get_campaign(cid):
        raise HTTPException(404, "no such campaign")
    store.set_campaign_status(cid, body.status)
    return store.get_campaign(cid)


@router.patch("/admin/campaigns/{cid}/requires", dependencies=[Depends(admin)])
def set_requires(cid: int, body: RequiresIn) -> dict:
    if not store.get_campaign(cid):
        raise HTTPException(404, "no such campaign")
    try:
        return store.set_requirement(cid, body.requires_campaign_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/admin/campaigns/{cid}", dependencies=[Depends(admin)])
def delete_campaign(cid: int) -> dict:
    if not store.get_campaign(cid):
        raise HTTPException(404, "no such campaign")
    try:
        store.delete_campaign(cid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@router.get("/admin/campaigns/{cid}/stats", dependencies=[Depends(admin)])
def campaign_stats(cid: int) -> dict:
    if not store.get_campaign(cid):
        raise HTTPException(404, "no such campaign")
    return store.stats(cid)


@router.get("/admin/campaigns/{cid}/disagreements", dependencies=[Depends(admin)])
def campaign_disagreements(cid: int) -> dict:
    return {"items": store.disagreements(cid)}


@router.get("/admin/campaigns/{cid}/q/{qid}", dependencies=[Depends(admin)])
def campaign_question(cid: int, qid: str) -> dict:
    try:
        return store.admin_question(cid, qid)
    except KeyError:
        raise HTTPException(404, "no such question")


@router.put("/admin/campaigns/{cid}/adjudicate/{qid}")
def adjudicate(cid: int, qid: str, body: AdjudicateIn, user: str = Depends(admin)) -> dict:
    try:
        return store.adjudicate(cid, qid, body.decision, body.note, (body.decided_by or "").strip() or user)
    except KeyError:
        raise HTTPException(404, "no such question")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/admin/campaigns/{cid}/export", dependencies=[Depends(admin)])
def export(cid: int, format: str = Query("json")):
    if not store.get_campaign(cid):
        raise HTTPException(404, "no such campaign")
    if format == "csv":
        return Response(store.export_csv(cid), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="review_{cid}.csv"'})
    return store.export_rows(cid)
