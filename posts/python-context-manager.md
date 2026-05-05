## 核心概念
`with` 语句负责进入和退出一段受管理的上下文。常见例子是文件、锁、数据库连接和临时配置。

## 示例
```python
from contextlib import contextmanager

@contextmanager
def section(name):
    print("enter", name)
    try:
        yield
    finally:
        print("exit", name)
```

## 判断
当一段逻辑必须成对执行 setup 和 cleanup 时，就应该考虑 context manager。
