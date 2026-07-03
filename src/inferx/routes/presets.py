"""Preset-related routes: CRUD and clone."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models import Preset
from . import get_config

router = APIRouter()


@router.get("/presets")
async def list_presets():
    return get_config().list_presets()


@router.post("/presets", response_model=Preset)
async def create_preset(body: Preset):
    return get_config().save_preset(body)


@router.put("/presets/{name}", response_model=Preset)
async def update_preset(name: str, body: Preset):
    body.name = name
    return get_config().save_preset(body)


@router.delete("/presets/{name}")
async def delete_preset(name: str):
    ok = get_config().delete_preset(name)
    if not ok:
        raise HTTPException(404, f"Preset not found: {name}")
    return {"success": True}


@router.post("/presets/{name}/clone")
async def clone_preset(name: str, new_name: str):
    """Clone a preset with a new name."""
    preset = get_config().get_preset(name)
    if not preset:
        raise HTTPException(404, f"Preset not found: {name}")

    new_preset = preset.model_copy(update={"name": new_name})
    return get_config().save_preset(new_preset)
