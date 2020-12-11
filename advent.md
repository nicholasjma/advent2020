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

from collections import Counter, defaultdict, deque
from copy import deepcopy
from functools import reduce, lru_cache
from itertools import count, combinations, chain
from IPython.display import clear_output
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
    l = f.read().splitlines()
    print(sum(map(valid2, l)))
```

## Day 3

```python
class cstr(str):
    """Cylindrical string class"""
    def __getitem__(self, key):
        """Cylindrical getitem"""
        # Note this doesn't quite work for slices so we just use the original logic
        try:
            return super().__getitem__(key % len(self))
        except TypeError:
            return super().__getitem__(key)

    def is_tree(self, idx, treechar="#"):
        """Return True if tree in position idx"""
        return self[idx] == treechar


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

Here's another solution using regex

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
    """Assembly machine for Advent of Code 2020"""
    def __init__(self, tape):
        # either interpret tape as a filename, or as a list
        if isinstance(tape, list):
            # avoid any mutation bugs
            self.inst = deepcopy(tape)
        else:
            with open(tape) as f:
                self.inst = [(x[:3], int(x[4:])) for x in f.read().splitlines()]
        self.reset()

    def run(self):
        """Run instructions until we either halt or loop"""
        while True:
            out = self.step()
            if out[0] != -1:
                return out

    def reset(self):
        self.cur, self.acc, self.ex = 0, 0, set()

    def step(self):
        if self.cur in self.ex:
            return (False, self.acc)
        self.ex.add(self.cur)
        op, num = self[self.cur]
        if op == "nop":
            self.cur += 1
        elif op == "jmp":
            self.cur += num
        elif op == "acc":
            self.acc += num
            self.cur += 1
        if self.cur == len(self):
            return True, self.acc
        elif self.cur > len(self):
            # we overshot the end
            return None, self.acc
        return -1, self.acc


    def replace(self, i, opmap):
        """Replace the instruction in position i based on opmap dict"""
        op, val = self[i]
        self[i] = (opmap.get(op, op), val)

    def __getitem__(self, index):
        """Return item(s) from the tape."""
        return self.inst[index]

    def __setitem__(self, index, value):
        """Set value in tape"""
        self.inst[index] = value

    def __len__(self):
        """Return current length of tape"""
        return len(self.inst)

    def _print(self, sl=None):
        """Return a string representation, optionally of a particular slice"""
        if sl is None:
            sl = slice()
        return "\n".join(
            f"{x} {'+' if y >= 0 else ''}{y}"
            for i, (x, y) in enumerate(self.inst[sl])
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
swap = {"nop": "jmp", "jmp": "nop"}
for i in range(len(a)):
    a.reset()
    # replace ith instruction. Note that nothing happens unless it's nop or jmp
    a.replace(i, swap)
    try:
        if (h := a.run())[0]:
            print(h)
            break
    finally:
        a.replace(i, swap)
```

## Day 9

```python
with open("input9.txt") as f:
    l = list(map(int, f.read().splitlines()))
recent, l = deque(l[:25]), l[25:]

for n in l:
    if not any(n - m in recent and n - m != m for m in recent):
        print(n)
        break
    recent.popleft()
    recent.append(n)
```

```python
with open("input9.txt") as f:
    l = list(map(int, f.read().splitlines()))
```

```python
m = deepcopy(l)
for i, x in enumerate(m[1:], start=1):
    m[i] += m[i - 1]
```

```python
mh = set(m)
target = 14144619
for idx, x in enumerate(m):
    if (s_t:=x + target) in mh:
        bounds = slice(idx + 1, m.index(s_t) + 1)
        break
min(l[bounds]) + max(l[bounds])
```

## Day 10

```python
with open("input10.txt") as f:
    l = sorted(list(map(int, f.read().splitlines())))
c = Counter(y - x for x, y in zip([0] + l, l + [max(l) + 3]))
c[1] * c[3]
```

Or we could use pandas

```python
l = pd.read_csv("input10.txt", header=None)[0].sort_values().to_list()
pd.Series([0] + l + [l[-1] + 3]).diff().value_counts().prod()
```

```python
with open("input10.txt") as f:
    l = [0] + sorted(list(map(int, f.read().splitlines())))
sol = [0] * len(l)
for i in range(-1, -len(l) - 1, -1):
    sol[i] = (i == -1) + sum(
        sol[j] for j in range(i + 1, 0) if l[i] + 3 >= l[j]
    )
sol[0]
```

```python
@lru_cache
def solution(n=0):
    return (n == len(l) - 1) + sum(
        solution(j) for j in range(n + 1, len(l)) if l[n] + 3 >= l[j]
    )
solution()
```

## Day 11

```python
class Grid:
    def __init__(self, l):
        if isinstance(l, list):
            self.l = l
        else:
            with open(l) as f:
                self.l = list(map(list, f.read().splitlines()))
        self.m = len(self.l)
        self.n = len(self.l[0])
        self.adj = [[self._adjacent(r, c) for c in range(self.n)] for r in range(self.m)]
        self.adj_changes = deque()
        

    DIRS = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1)
    )

    def __getitem__(self, loc):
        r, c = loc
        return self.l[r][c]

    def __setitem__(self, loc, value):
        row, col = loc
        old = self.l[row][col]
        self.l[row][col] = value
        if old == "L" and value == "#":
            update = 1
        elif old == "#" and value == "L":
            update = -1
        else:
            return
        for x, y in self.DIRS:
            r, c = row + x, col + y
            if not (0 <= r < self.m and 0 <= c < self.n):
                continue
            else:
                self.adj_changes.append((r, c, update))
                
    def _adjacent(self, row, col):
        out = 0
        for x, y in Grid.DIRS:
            r, c = row + x, col + y
            if not (0 <= r < self.m and 0 <= c < self.n):
                continue
            out += self[r,c] == "#"
        return out

    def update_adj(self):
        while len(self.adj_changes) > 0:
            r, c, update = self.adj_changes.pop()
            self.adj[r][c] += update

    def adjacent(self, row, col):
        return self.adj[row][col]

    def occupied(self):
        return sum(self[r, c] == "#" for r in range(self.n) for c in range(self.n))

    def print(self):
        clear_output(wait=True)
        print("\n".join("".join(x) for x in self.l))

    def evolve(self, threshold=4):
        changed = False
        for r in range(self.m):
            for c in range(self.n):
                o = self.adjacent(r, c)
                if self[r, c] == "L" and o == 0:
                    changed = True
                    self[r, c] = "#"
                elif self[r, c] == "#" and o >= threshold:
                    changed = True
                    self[r, c] = "L"
        self.update_adj()
        return changed
```

```python
g = Grid("input11.txt")
while g.evolve():
    g.print()

print(g.occupied())
```

```python code_folding=[6]
class Grid2(Grid):
    def _adjacent(self, row, col):
        out = 0
        for x, y in self.DIRS:
            r, c = row + x, col + y
            while (valid:=(0 <= r < self.m and 0 <= c < self.n)) and self.l[r][c] == ".":
                r, c = r + x, c + y
            if valid and self.l[r][c] == "#":
                out += 1
        return out

    def evolve(self, threshold=5):
        return super().evolve(threshold)

    def __setitem__(self, loc, value):
        row, col = loc
        old = self[row, col]
        self.l[row][col] = value
        if old == "L" and value == "#":
            update = 1
        elif old == "#" and value == "L":
            update = -1
        else:
            return
        for x, y in self.DIRS:
            r, c = row + x, col + y
            while (valid:=(0 <= r < self.m and 0 <= c < self.n)) and self.l[r][c] == ".":
                r, c = r + x, c + y
            if valid:
                self.adj_changes.append((r, c, update))

g = Grid2("input11.txt")
while g.evolve():
    g.print()
print(g.occupied())
```
