---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd
import re
import math
import operator
from functools import reduce
import yaml
from itertools import count, combinations
import attr
```

## Day 1

```python
with open("input.txt") as f:
    l = [int(x) for x in f.read().split("\n")[:-1]]
```

```python
def get_combo(l, s, k):
    l = set(l)
    for pair in combinations(l,  k - 1):
        if s - sum(pair) in l:
            return np.prod([*pair, s - sum(pair)])
```

```python
get_combo(l, 2020, 2)
```

```python
get_combo(l, 2020, 3)
```

## Day 2

```python
def valid(s):
    m = re.match(r"(\d+)-(\d+) ([a-z]): (.*)", s)
    a, b, l, s = m.groups()
    a, b = int(a), int(b)
    n = sum(c == l for c in s)
    return a <= n <= b

with open("input2.txt") as f:
    l = f.read().split("\n")[:-1]
    print(sum(map(valid, l)))
```

```python
def valid2(s):
    m = re.match(r"(\d+)-(\d+) ([a-z]): (.*)", s)
    a, b, l, s = m.groups()
    a, b = int(a), int(b)
    return (s[a - 1] == l) ^ (s[b - 1] == l)

with open("input2.txt") as f:
    l = f.read().split("\n")[:-1]
    print(sum(map(valid2, l)))
```

## Day 3

```python
class cstr(str):
    def __getitem__(self, key):
        return super().__getitem__(key % len(self))
    def is_tree(self, idx):
        return self[idx] == "#"

def trees(l, d, r):
    return sum(l[d * i].is_tree(r * i) for i in range(len(l)) if i * d < len(l))
```

```python
with open("input3.txt") as f:
    l = [cstr(x) for x in f.read().split("\n") if x != ""]
trees(l, 1, 3)
```

```python
with open("input3.txt") as f:
    l = [cstr(x) for x in f.read().split("\n") if x != ""]
slopes = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
np.prod([trees(l, d, r) for r, d in slopes])
```

## Day 4

```python
@attr.s
class Passport:
    byr = attr.ib()
    iyr = attr.ib()
    eyr = attr.ib()
    hgt = attr.ib()
    hcl = attr.ib()
    ecl = attr.ib()
    pid = attr.ib()
    cid = attr.ib(default="")
out = 0
with open("input4.txt") as f:
    lines = f.read()[:-1].split("\n\n")
    for l in lines:
        l = (l + "\n").replace("\n", " ").replace(" ", '"\n').replace(":", ': "')
        l = "---\n" + l
        strs = yaml.safe_load(l)
        try:
            Passport(**strs)
        except (TypeError, ValueError):
            pass
        else:
            out += 1
out
```

```python
@attr.s
class Passport:
    byr = attr.ib()
    iyr = attr.ib()
    eyr = attr.ib()
    hgt = attr.ib()
    hcl = attr.ib()
    ecl = attr.ib()
    pid = attr.ib()
    cid = attr.ib(default="")

    @byr.validator
    def byr_validator(self, attribute, value):
        if len(value) != 4 or not 1920 <= int(value) <= 2002:
            raise ValueError

    @iyr.validator
    def iyr_validator(self, attribute, value):
        if len(value) != 4 or not 2010 <= int(value) <= 2020:
            raise ValueError

    @eyr.validator
    def eyr_validator(self, attribute, value):
        if len(value) != 4 or not 2020 <= int(value) <= 2030:
            raise ValueError

    @hgt.validator
    def hgt_validator(self, attribute, value):
        h, units = int(value[:-2]), value[-2:]
        if units == "cm":
            if not 150 <= h <= 193:
                raise ValueError
        elif units == "in":
            if not 59 <= h <= 76:
                raise ValueError
        else:
            raise ValueError

    @hcl.validator
    def hcl_validator(self, attribute, value):
        if not bool(re.match("#[0-9a-f]{6}", value)):
            raise ValueError

    @ecl.validator 
    def ecl_validator(self, attribute, value):
        if value not in {"amb", "blu", "brn", "gry", "grn", "hzl", "oth"}:
            raise ValueError

    @pid.validator
    def pid_validator(self, attribute, value):
        if not re.match("[0-9]{9}$", value):
            raise ValueError

out = 0
with open("input4.txt") as f:
    lines = f.read()[:-1].split("\n\n")
    for l in lines:
        l = (l + "\n").replace("\n", " ").replace(" ", '"\n').replace(":", ': "')
        l = "---\n" + l
        strs = yaml.safe_load(l)
        try:
            Passport(**strs)
        except (TypeError, ValueError):
            pass
        else:
            out += 1
out
```

Here's another solution using regex for part 2

```python
REGEX = [
    r"byr:(?:19[2-9]\d|200[0-2])\b",
    r"iyr:20(?:1\d|20)\b",
    r"eyr:20(?:2\d|30)\b",
    r"hgt:(?:1(?:[5-8]\d|9[0-3])cm|(?:59|6\d|7[0-6])in)\b",
    r"hcl:#[0-9a-f]{6}\b",
    r"ecl:(amb|blu|brn|gry|grn|hzl|oth)\b",
    r"pid:\d{9}\b",
]
with open("input4.txt") as f:
    lines = f.read()[:-1].split("\n\n")
    print(sum(all(re.search(x, s) for x in REGEX) for s in lines))
```

## Day 5

```python
def binsum(s, true):
    out = 0
    for x in s:
        out <<= 1
        out += x in true
    return out
    
with open("input5.txt") as f:
    seats = [binsum(s, "BR") for s in f.read().split("\n")[:-1]]
max(seats)
```

```python
seats = sorted(seats)
min(a for a, b in zip(seats, seats[1:]) if b - a > 1)
```
