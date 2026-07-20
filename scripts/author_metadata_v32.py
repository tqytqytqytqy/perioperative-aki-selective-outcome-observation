#!/usr/bin/env python3
"""Single source of truth for confirmed v3.2 author metadata."""

from __future__ import annotations


AUTHOR_METADATA = [
    {
        "order": 1,
        "name": "Qingyu Teng",
        "given_names": "Qingyu",
        "family_names": "Teng",
        "affiliation": "1",
        "equal_contribution": True,
        "corresponding": False,
        "email": "",
        "orcid": "",
    },
    {
        "order": 2,
        "name": "Yingya Zhao",
        "given_names": "Yingya",
        "family_names": "Zhao",
        "affiliation": "2",
        "equal_contribution": True,
        "corresponding": False,
        "email": "",
        "orcid": "",
    },
    {
        "order": 3,
        "name": "Jin Zhao",
        "given_names": "Jin",
        "family_names": "Zhao",
        "affiliation": "1",
        "equal_contribution": False,
        "corresponding": False,
        "email": "",
        "orcid": "",
    },
    {
        "order": 4,
        "name": "Qian Chen",
        "given_names": "Qian",
        "family_names": "Chen",
        "affiliation": "1",
        "equal_contribution": False,
        "corresponding": False,
        "email": "",
        "orcid": "",
    },
    {
        "order": 5,
        "name": "Min Tao",
        "given_names": "Min",
        "family_names": "Tao",
        "affiliation": "1",
        "equal_contribution": False,
        "corresponding": False,
        "email": "",
        "orcid": "",
    },
    {
        "order": 6,
        "name": "Qi Li",
        "given_names": "Qi",
        "family_names": "Li",
        "affiliation": "1",
        "equal_contribution": False,
        "corresponding": True,
        "email": "",
        "orcid": "0009-0003-3140-5887",
    },
    {
        "order": 7,
        "name": "Tao Xu",
        "given_names": "Tao",
        "family_names": "Xu",
        "affiliation": "1",
        "equal_contribution": False,
        "corresponding": True,
        "email": "",
        "orcid": "0000-0001-5868-4079",
    },
    {
        "order": 8,
        "name": "Hui Zhang",
        "given_names": "Hui",
        "family_names": "Zhang",
        "affiliation": "1",
        "equal_contribution": False,
        "corresponding": True,
        "email": "",
        "orcid": "0009-0006-5460-3845",
    },
]

AFFILIATIONS = {
    "1": (
        "Department of Anesthesiology, Shanghai Sixth People's Hospital, Shanghai Jiao Tong "
        "University School of Medicine, Shanghai, China"
    ),
    "2": "School of Public Health, Fudan University, Shanghai, China",
}

AUTHOR_NAMES = ", ".join(author["name"] for author in AUTHOR_METADATA)
CORE_PROPERTY_AUTHORS = "; ".join(author["name"] for author in AUTHOR_METADATA)
LICENSE_HOLDER_NAMES = ", ".join(author["name"] for author in AUTHOR_METADATA[:-1]) + ", and " + AUTHOR_METADATA[-1]["name"]


def byline() -> str:
    values = []
    for author in AUTHOR_METADATA:
        marker = ""
        if author["equal_contribution"]:
            marker += "†"
        if author["corresponding"]:
            marker += "*"
        values.append(f"{author['name']}{author['affiliation']}{marker}")
    return ", ".join(values)


def corresponding_authors() -> list[dict[str, object]]:
    return [author for author in AUTHOR_METADATA if author["corresponding"]]


def equal_contributors() -> list[str]:
    return [str(author["name"]) for author in AUTHOR_METADATA if author["equal_contribution"]]


def validate_author_metadata() -> None:
    names = [str(author["name"]) for author in AUTHOR_METADATA]
    orders = [int(author["order"]) for author in AUTHOR_METADATA]
    if len(names) != len(set(names)):
        raise ValueError("Author names must be unique")
    if orders != list(range(1, len(AUTHOR_METADATA) + 1)):
        raise ValueError("Author order must be contiguous and one-based")
    if set(str(author["affiliation"]) for author in AUTHOR_METADATA) - set(AFFILIATIONS):
        raise ValueError("Every author affiliation must resolve")
    corresponding = corresponding_authors()
    email_presence = [bool(str(author["email"]).strip()) for author in corresponding]
    if any(email_presence) and not all(email_presence):
        raise ValueError("Corresponding author emails must be either complete or fully redacted")
    for author in corresponding:
        email = str(author["email"]).strip()
        if email and "@" not in email:
            raise ValueError(f"Corresponding author email is malformed: {author['name']}")
        if not author["orcid"]:
            raise ValueError(f"Corresponding author ORCID missing: {author['name']}")
    if len(equal_contributors()) != 2:
        raise ValueError("Exactly two equal contributors are expected")


validate_author_metadata()
