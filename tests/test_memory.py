from app.memory.service import PersonalMemoryService


def test_save_and_search_personal_memory(db):
    service = PersonalMemoryService(db)
    service.save(
        memory_type="project",
        content="用户正在开发 EvoMe，它是长期 Personal AI 项目",
        importance=0.9,
    )
    results = service.search("EvoMe 是什么项目", top_k=3)
    assert results
    assert "EvoMe" in results[0].content


def test_personal_memory_duplicate(db):
    service = PersonalMemoryService(db)
    content = "用户偏好先讨论架构，再进入代码实现"
    service.save(memory_type="preference", content=content, importance=0.8)
    assert service.is_duplicate(content) is True
