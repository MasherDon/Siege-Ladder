"""Prompts of the server benchmarks, kept exactly as measured (two of them are in Russian, our working language)."""

CODE = '''class TTLCache:
    """LRU-кэш с временем жизни записи."""

    def __init__(self, capacity, ttl):
        self.capacity = capacity
        self.ttl = ttl
        self.data = {}      # key -> (value, expires)
        self.order = []     # от старых к новым

    def get(self, key, now):
        if key not in self.data:
            return None
        value, expires = self.data[key]
        if expires < now:
            del self.data[key]
            return None
        self.order.remove(key)
        self.order.append(key)
        return value

    def put(self, key, value, now):
        if key in self.data:
            self.order.remove(key)
        elif len(self.data) >= self.capacity:
            oldest = self.order.pop(0)
            del self.data[oldest]
        self.data[key] = (value, now + self.ttl)
        self.order.append(key)

    def purge(self, now):
        for key in list(self.data):
            if self.data[key][1] < now:
                del self.data[key]
'''

PROMPTS = {
    "ru-prose": "Ты ведущий вечернего эфира музыкальной радиостанции. Расскажи слушателям подробно, в двадцати–"
                "двадцати пяти фразах, что будет в эфире сегодня вечером: музыкальные блоки, гости, рубрики, "
                "новости и чем закончится вечер.",
    "ru-code": "Вот класс на Python:\n```python\n" + CODE + "```\nОбъясни подробно, что делает каждый метод, найди "
               "ошибки (их несколько) и приведи исправленный код целиком.",
    "en-tech": "Explain in detail how a mixture-of-experts transformer routes each token to experts, what top-k "
               "routing and load balancing are, and why keeping frequently used experts in GPU memory speeds up "
               "single-user decoding on a PC with limited VRAM. Use several paragraphs.",
    # an English code task in place of a private one of ours
    "en-code": "Here is a Python class:\n```python\n" + CODE + "```\nExplain in detail what each method does, find the "
               "bugs (there are several) and give the whole corrected code.",
}

CEIL = "Назови три основных цвета."   # "Name the three primary colours" - an 8-token answer: decode with no context
