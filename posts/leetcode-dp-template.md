## 记录结构
刷动态规划时，不要一上来写代码。先把题目转成几个稳定问题：

- 状态表示什么？
- 最后一步是什么？
- 转移来自哪里？
- 初始化和遍历顺序是什么？

## Python 模板
```python
def solve(nums):
    dp = [0] * len(nums)
    for i, value in enumerate(nums):
        dp[i] = value
    return max(dp)
```

## 复盘
每道题最后补一句：这题和哪类题相似，下一次应该从哪个角度识别。
