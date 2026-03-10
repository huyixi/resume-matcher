from app.services.refiner import validate_master_alignment


def test_alignment_allows_terms_supported_by_master_body_text() -> None:
    master = {
        "summary": "",
        "workExperience": [
            {
                "description": [
                    "负责性能排查，展现了前端性能优化和浏览器兼容性处理能力",
                ]
            }
        ],
        "personalProjects": [],
        "additional": {"technicalSkills": [], "certificationsTraining": []},
    }
    tailored = {
        "additional": {
            "technicalSkills": ["前端性能优化", "浏览器兼容性"],
            "certificationsTraining": [],
        }
    }

    report = validate_master_alignment(tailored, master)

    assert report.is_aligned is True
    assert report.violations == []


def test_alignment_matches_common_framework_aliases() -> None:
    master = {
        "summary": "Built internal tools with Vue 3 and Element Plus.",
        "workExperience": [],
        "personalProjects": [],
        "additional": {"technicalSkills": [], "certificationsTraining": []},
    }
    tailored = {
        "additional": {
            "technicalSkills": ["Vue.js", "Element-ui"],
            "certificationsTraining": [],
        }
    }

    report = validate_master_alignment(tailored, master)

    assert report.is_aligned is True
    assert report.violations == []


def test_alignment_blocks_terms_missing_from_master_resume() -> None:
    master = {
        "summary": "Frontend engineer focused on Vue and design systems.",
        "workExperience": [],
        "personalProjects": [],
        "additional": {"technicalSkills": ["Vue 3"], "certificationsTraining": []},
    }
    tailored = {
        "additional": {
            "technicalSkills": ["Vue", "React"],
            "certificationsTraining": [],
        }
    }

    report = validate_master_alignment(tailored, master)

    assert report.is_aligned is False
    assert [violation.value for violation in report.violations] == ["React"]
