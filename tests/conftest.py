"""Shared test fixtures for codebase-intel."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a minimal sample Python project for integration tests."""
    src = tmp_path / "src"
    src.mkdir()

    # models.py — base module
    (src / "models.py").write_text(
        '''"""Data models."""

class User:
    """A user in the system."""

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    def display_name(self) -> str:
        return f"{self.name} <{self.email}>"


class Order:
    """A purchase order."""

    def __init__(self, user: User, total: float) -> None:
        self.user = user
        self.total = total
'''
    )

    # service.py — imports models
    (src / "service.py").write_text(
        '''"""Business logic."""

from models import User, Order


class UserService:
    """Manages user operations."""

    async def get_user(self, user_id: int) -> User:
        return User(name="test", email="test@example.com")

    async def create_order(self, user: User, amount: float) -> Order:
        return Order(user=user, total=amount)
'''
    )

    # api.py — imports service
    (src / "api.py").write_text(
        '''"""API endpoints."""

from service import UserService

service = UserService()


async def get_user_endpoint(user_id: int):
    """Get a user by ID."""
    user = await service.get_user(user_id)
    return {"name": user.display_name()}
'''
    )

    # tests/test_service.py — test file
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_service.py").write_text(
        '''"""Tests for the service module."""

from service import UserService


async def test_get_user():
    svc = UserService()
    user = await svc.get_user(1)
    assert user.name == "test"
'''
    )

    return tmp_path


@pytest.fixture
def sample_js_project(tmp_path: Path) -> Path:
    """Create a minimal sample JS/TS project for integration tests."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "utils.ts").write_text(
        '''export function formatDate(date: Date): string {
    return date.toISOString();
}

export function capitalize(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
'''
    )

    (src / "api.ts").write_text(
        '''import { formatDate } from "./utils";

export interface User {
    id: number;
    name: string;
    createdAt: Date;
}

export async function fetchUser(id: number): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}
'''
    )

    (src / "index.ts").write_text(
        '''import { fetchUser } from "./api";
import { capitalize } from "./utils";

export { fetchUser, capitalize };
'''
    )

    return tmp_path
