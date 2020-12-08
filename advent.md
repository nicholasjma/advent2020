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
    display_name: Python [conda env:py38]
    language: python
    name: conda-env-py38-py
---

```python
import math
import operator
import re

import attr
import numpy as np
import pandas as pd
import yaml

from collections import Counter, defaultdict
from copy import deepcopy
from functools import reduce
from itertools import count, combinations
```

## Day 1

```python
with open("input.txt") as f:
    l = [int(x) for x in f.read().splitlines()]
```

```python
def get_combo(l, s, k):
    l = set(l)
    for pair in combinations(l, k - 1):
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
    a, b, l, s = re.split(": |-| ", s)
    a, b = int(a), int(b)
    n = sum(c == l for c in s)
    return a <= n <= b


with open("input2.txt") as f:
    l = f.read().splitlines()
    print(sum(map(valid, l)))
```

```python
def valid2(s):
    a, b, l, s = re.split(": |-| ", s)
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
    l = [cstr(x) for x in f.read().splitlines()]
trees(l, 1, 3)
```

```python
with open("input3.txt") as f:
    l = [cstr(x) for x in f.read().splitlines()]
slopes = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
np.prod([trees(l, d, r) for r, d in slopes])
```

## Day 4


Here are solutions using attrs validators

```python
@attr.s
class Passport:
    """Passport class, will error if required field is missing"""
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
        # coerce input to yaml format
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

```python
# make regex dictionary from previous regexes
REGEX2 = {x[:3]: x[4:] for x in REGEX}


@attr.s
class Passport:
    def validator(self, attribute, value):
        if not re.match(REGEX2[attribute.name], value):
            raise ValueError

    byr = attr.ib(validator=validator)
    iyr = attr.ib(validator=validator)
    eyr = attr.ib(validator=validator)
    hgt = attr.ib(validator=validator)
    hcl = attr.ib(validator=validator)
    ecl = attr.ib(validator=validator)
    pid = attr.ib(validator=validator)
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

## Day 5

```python
def binsum(s, true):
    """Convert binary to integer, with true the set of characters that are 1"""
    out = 0
    for x in s:
        out <<= 1
        out += x in true
    return out


with open("input5.txt") as f:
    seats = [binsum(s, "BR") for s in f.read().splitlines()]
max(seats)
```

```python
seats = sorted(seats)
min(a for a, b in zip(seats, seats[1:]) if b - a > 1)
```

## Day 6

```python
with open("input6.txt") as f:
    l = f.read()[:-1].split("\n\n")
```

```python
sum(len(set.union(set(group) - {"\n"})) for group in l)
```

```python
sum(len(set.intersection(*map(set, group.splitlines()))) for group in l)
```

Using `collections.Counter`

```python
sum(x != "\n" for c in map(Counter, l) for x in c)
```

```python
sum(c[x] > c.get("\n", 0) for c in map(Counter, l) for x in c)
```

## Day 7

```python
with open(r"input7.txt") as f:
    d = f.read().splitlines()

contains = defaultdict(list)
contained = defaultdict(set)
for x in d:
    color = re.match("^([\w ]*) bags contain", x)[1]
    for num, inner_color in re.findall(r"(\d+) (.+?) bags?[,.]", x):
        # update contained and contains with data
        contained[inner_color].add(color)
        contains[color].append((int(num), inner_color))
```

```python code_folding=[]
def update_good(color):
    """Update good set with bags that contain color"""
    for col in contained[color]:
        good.add(col)
        update_good(col)

good = set()
update_good("shiny gold")
len(good)
```

```python
def count_bags(color):
    """Recursively count bags contained in bag of specified color"""
    out = 0
    for num, inner_color in contains[color]:
        out += num + num * count_bags(inner_color)
    return out


count_bags("shiny gold")
```

## Day 8

```python
class Assembly:
    """Assembly machine for Advent of Code 2019"""
    def __init__(self, tape):
        # either interpret tape as a filename, or as a list
        if isinstance(tape, list):
            # avoid any mutation bugs
            self.inst = deepcopy(tape)
        else:
            with open(tape) as f:
                self.inst = [(x[:3], int(x[4:])) for x in f.read().splitlines()]

    def run(self):
        """Run instructions until we either halt or loop"""
        cur, acc, ex = 0, 0, set()
        while cur not in ex:
            ex.add(cur)
            op, num = self[cur]
            if op == "nop":
                cur += 1
            elif op == "jmp":
                cur += num
            elif op == "acc":
                acc += num
                cur += 1
            if cur == len(self):
                return True, acc
            elif cur > len(self):
                # we overshot the end
                return False, acc
        # if we get here, we looped
        return False, acc

    def replace(self, i, opmap):
        """Replace the instruction in position i based on opmap dict"""
        op, val = self[i]
        self[i] = (opmap.get(op, op), val)

    def __getitem__(self, index):
        """Return item from the tape. Return a new Assembly object if a slice."""
        if isinstance(index, slice):
            return Assembly(self.inst[index])
        return self.inst[index]

    def __setitem__(self, index, value):
        """Set value in tape"""
        self.inst[index] = value

    def __len__(self):
        """Return current length of tape"""
        return len(self.inst)

    def _print(self, sl=None):
        """Return a string representation, optionally of a particular slice"""
        if sl is not None:
            return "\n".join(
                f"{x} {'+' if y >= 0 else ''}{y}"
                for i, (x, y) in enumerate(self.inst[sl])
            )
        else:
            return "\n".join(
                f"{x} {'+' if y >= 0 else ''}{y}" for i, (x, y) in enumerate(self.inst)
            )

    def print(self, sl=None):
        """Print the string representation"""
        print(self._print(sl))

    def __repr__(self):
        """Return the string representation of the whole tape"""
        return self._print()
```

```python
a = Assembly("input8.txt")
a.run()
```

```python
a = Assembly("input8.txt")
swap = {"nop": "jmp", "jmp": "nop"}
for i in range(len(a)):
    a.replace(i, swap)
    if (h := a.run())[0]:
        print(h)
        break
    a.replace(i, swap)
```
