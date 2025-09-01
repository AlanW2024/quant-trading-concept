## 關聯議題/備註
<!-- 請填寫此 PR 相關的議題 (e.g., Fixes #123) 或其他備註。 -->


---

## 接受標準對照
<!-- 請對照 Planner 的接受標準，逐項確認。 -->
- [ ] **目標**：
- [ ] **產物/檔案**：
- [ ] **邊界**：

---

## 風險與回滾
- **風險**：<!-- 此變更可能帶來的風險，例如效能影響、資料庫遷移、破壞性 API 變更等。 -->
- **回滾**：<!-- 如果出現問題，如何安全地回滾此變更？例如：`git revert <commit_hash>` -->

---

## 本機檢查清單 (Local Checks)
**重要**: 提交前，請確保所有本地檢查均已通過。

- [ ] `black --check .`
- [ ] `ruff check .`
- [ ] `mypy .`
- [ ] `python -m pytest -q`

---

## Pro-only 政策提醒
*   `run_engine.py` **必須**保持 Pro-only 邏輯，找不到 Pro 引擎應直接報錯。
*   與 Pro 引擎相關的測試，在缺少 `professional_multifactor_engine_pro.py` 的環境中**必須**被 `skip`，不得 `fail`。