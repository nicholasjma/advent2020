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

## Imports

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
from itertools import count, combinations, chain, product
from IPython.display import clear_output
```

<!-- #region heading_collapsed=true -->
## Day 1
<!-- #endregion -->

```python hidden=true
with open("input.txt") as f:
    l = [int(x) for x in f.read().splitlines()]
```

```python hidden=true
def get_combo(l, s, k):
    l = set(l)
    for pair in combinations(l, k - 1):
        if s - sum(pair) in l:
            return np.prod([*pair, s - sum(pair)])
```

```python hidden=true
get_combo(l, 2020, 2)
```

```python hidden=true
get_combo(l, 2020, 3)
```

<!-- #region heading_collapsed=true -->
## Day 2
<!-- #endregion -->

```python hidden=true
def valid(s):
    a, b, l, s = re.split(": |-| ", s)
    a, b = int(a), int(b)
    n = sum(c == l for c in s)
    return a <= n <= b


with open("input2.txt") as f:
    l = f.read().splitlines()
    print(sum(map(valid, l)))
```

```python hidden=true
def valid2(s):
    a, b, l, s = re.split(": |-| ", s)
    a, b = int(a), int(b)
    return (s[a - 1] == l) ^ (s[b - 1] == l)


with open("input2.txt") as f:
    l = f.read().splitlines()
    print(sum(map(valid2, l)))
```

<!-- #region heading_collapsed=true -->
## Day 3
<!-- #endregion -->

```python hidden=true
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

```python hidden=true
with open("input3.txt") as f:
    l = [cstr(x) for x in f.read().splitlines()]
trees(l, 1, 3)
```

```python hidden=true
with open("input3.txt") as f:
    l = [cstr(x) for x in f.read().splitlines()]
slopes = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
np.prod([trees(l, d, r) for r, d in slopes])
```

<!-- #region heading_collapsed=true -->
## Day 4
<!-- #endregion -->

<!-- #region hidden=true -->
Here are solutions using attrs validators
<!-- #endregion -->

```python hidden=true
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

```python hidden=true
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

<!-- #region hidden=true -->
Here's another solution using regex
<!-- #endregion -->

```python hidden=true
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

```python hidden=true
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

<!-- #region heading_collapsed=true -->
## Day 5
<!-- #endregion -->

```python hidden=true
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

```python hidden=true
seats = sorted(seats)
min(a for a, b in zip(seats, seats[1:]) if b - a > 1)
```

<!-- #region heading_collapsed=true -->
## Day 6
<!-- #endregion -->

```python hidden=true
with open("input6.txt") as f:
    l = f.read()[:-1].split("\n\n")
```

```python hidden=true
sum(len(set.union(set(group) - {"\n"})) for group in l)
```

```python hidden=true
sum(len(set.intersection(*map(set, group.splitlines()))) for group in l)
```

<!-- #region hidden=true -->
Using `collections.Counter`
<!-- #endregion -->

```python hidden=true
sum(x != "\n" for c in map(Counter, l) for x in c)
```

```python hidden=true
sum(c[x] > c.get("\n", 0) for c in map(Counter, l) for x in c)
```

<!-- #region heading_collapsed=true -->
## Day 7
<!-- #endregion -->

```python hidden=true
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

```python code_folding=[] hidden=true
def update_good(color):
    """Update good set with bags that contain color"""
    for col in contained[color]:
        good.add(col)
        update_good(col)


good = set()
update_good("shiny gold")
len(good)
```

```python hidden=true
def count_bags(color):
    """Recursively count bags contained in bag of specified color"""
    out = 0
    for num, inner_color in contains[color]:
        out += num + num * count_bags(inner_color)
    return out


count_bags("shiny gold")
```

<!-- #region heading_collapsed=true -->
## Day 8
<!-- #endregion -->

```python hidden=true
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
            f"{x} {'+' if y >= 0 else ''}{y}" for i, (x, y) in enumerate(self.inst[sl])
        )

    def print(self, sl=None):
        """Print the string representation"""
        print(self._print(sl))

    def __repr__(self):
        """Return the string representation of the whole tape"""
        return self._print()
```

```python hidden=true
a = Assembly("input8.txt")
a.run()
```

```python hidden=true
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

<!-- #region heading_collapsed=true -->
## Day 9
<!-- #endregion -->

```python hidden=true
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

```python hidden=true
with open("input9.txt") as f:
    l = list(map(int, f.read().splitlines()))
```

```python hidden=true
m = deepcopy(l)
for i, x in enumerate(m[1:], start=1):
    m[i] += m[i - 1]
```

```python hidden=true
mh = set(m)
target = 14144619
for idx, x in enumerate(m):
    if (s_t := x + target) in mh:
        bounds = slice(idx + 1, m.index(s_t) + 1)
        break
min(l[bounds]) + max(l[bounds])
```

<!-- #region heading_collapsed=true -->
## Day 10
<!-- #endregion -->

```python hidden=true
with open("input10.txt") as f:
    l = sorted(list(map(int, f.read().splitlines())))
c = Counter(y - x for x, y in zip([0] + l, l + [max(l) + 3]))
c[1] * c[3]
```

<!-- #region hidden=true -->
Or we could use pandas
<!-- #endregion -->

```python hidden=true
l = pd.read_csv("input10.txt", header=None)[0].sort_values().to_list()
pd.Series([0] + l + [l[-1] + 3]).diff().value_counts().prod()
```

```python hidden=true
with open("input10.txt") as f:
    l = [0] + sorted(list(map(int, f.read().splitlines())))
sol = [0] * len(l)
for i in range(-1, -len(l) - 1, -1):
    sol[i] = (i == -1) + sum(sol[j] for j in range(i + 1, 0) if l[i] + 3 >= l[j])
sol[0]
```

```python hidden=true
@lru_cache
def solution(n=0):
    return (n == len(l) - 1) + sum(
        solution(j) for j in range(n + 1, len(l)) if l[n] + 3 >= l[j]
    )


solution()
```

<!-- #region heading_collapsed=true -->
## Day 11
<!-- #endregion -->

```python hidden=true
class Grid:
    def __init__(self, l, threshold=4):
        if isinstance(l, list):
            self.l = l
        else:
            with open(l) as f:
                self.l = list(map(list, f.read().splitlines()))
        self.m = len(self.l)
        self.n = len(self.l[0])
        self.adj = [
            [self._adjacent(r, c) for c in range(self.n)] for r in range(self.m)
        ]
        self.adj_changes = defaultdict(lambda: 0)
        self.affected = {
            (r, c) for c in range(self.n) for r in range(self.m) if self[r, c] != "."
        }
        self.steps = 0
        self.threshold = threshold

    DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    FORMAT = {
        "#": "\033[32m█",
        "L": " ",
        ".": "\033[37m█",
    }

    def __getitem__(self, loc):
        r, c = loc
        return self.l[r][c]

    def __setitem__(self, loc, value):
        row, col = loc
        old = self.l[row][col]
        self.l[row][col] = value
        self.affected.add(loc)
        if old == "L" and value == "#":
            update = 1
        elif old == "#" and value == "L":
            update = -1
        else:
            return
        for x, y in self.DIRS:
            r, c = row + x, col + y
            if not (0 <= r < self.m) or not (0 <= c < self.n):
                continue
            else:
                self.adj_changes[r, c] += update
                new_adj = self.adj[r][c] + self.adj_changes[r, c]
                self.affected.add((r, c))

    def _adjacent(self, row, col):
        if self[row, col] == ".":
            return 0
        out = 0
        for x, y in Grid.DIRS:
            r, c = row + x, col + y
            if not (0 <= r < self.m and 0 <= c < self.n):
                continue
            out += self[r, c] == "#"
        return out

    def update_adj(self):
        for (r, c), v in self.adj_changes.items():
            self.adj[r][c] += v
        self.adj_changes = defaultdict(lambda: 0)

    def adjacent(self, row, col):
        return self.adj[row][col]

    def occupied(self):
        return sum(self[r, c] == "#" for r in range(self.n) for c in range(self.n))

    def print(self, override=False):
        if override or self.steps % 2 == 0:
            clear_output(wait=True)
            print(f"Generation {self.steps}    Occupied {self.occupied()}")
            print(
                "\033[32;40m"
                + "\n".join("".join(map(self.FORMAT.__getitem__, x)) for x in self.l)
            )

    def evolve(self):
        to_check = self.affected
        self.affected = set()
        for r, c in to_check:
            o = self.adjacent(r, c)
            if self[r, c] == "L" and o == 0:
                self[r, c] = "#"
            elif self[r, c] == "#" and o >= self.threshold:
                self[r, c] = "L"
        self.update_adj()
        self.steps += 1
        return len(self.affected) > 0


g = Grid("input11.txt", 4)
while g.evolve():
    g.print()
```

```python code_folding=[6] hidden=true
class Grid2(Grid):
    def __init__(self, l, threshold=5):
        self.vis = {}
        super().__init__(l, threshold)

    def _adjacent(self, row, col):
        if self[row, col] == ".":
            return 0
        out = 0
        for direction in self.DIRS:
            vis = self.get_vis(row, col, direction)
            if vis is not None:
                out += self[vis] == "#"
        return out

    def __setitem__(self, loc, value):
        row, col = loc
        old = self[row, col]
        self.l[row][col] = value
        self.affected.add(loc)
        if old == "L" and value == "#":
            update = 1
        elif old == "#" and value == "L":
            update = -1
        else:
            return
        for direction in self.DIRS:
            vis = self.get_vis(row, col, direction)
            if vis is not None:
                self.adj_changes[vis] += update

    #                 self.affected.add(vis)

    def get_vis(self, row, col, direction):
        if (row, col, direction) in self.vis:
            return self.vis[(row, col, direction)]
        x, y = direction
        r, c = row + x, col + y
        while (0 <= r < self.m and 0 <= c < self.n) and self.l[r][c] == ".":
            r, c = r + x, c + y
        if 0 <= r < self.m and 0 <= c < self.n:
            # cache results
            self.vis[(row, col, direction)] = (r, c)
            # if we are the first seat in a directionk, the reciprocal is true
            self.vis[(r, c, (-direction[0], -direction[1]))] = (row, col)
            return r, c
        else:
            return None


g = Grid2("input11.txt")
while g.evolve():
    g.print()
```

<!-- #region heading_collapsed=true -->
## Day 12
<!-- #endregion -->

```python hidden=true
with open("input12.txt") as f:
    l = f.read().splitlines()
```

```python hidden=true
px, py = 0, 0
DIRS = "ESWN"
DIR = ((1, 0), (0, -1), (-1, 0), (0, 1))
ROT = dict(zip("LR", (-1, 1)))
DIRS = dict(zip(DIRS, DIR))
facing = 0
for x in l:
    d, num = x[0], int(x[1:])
    if d in "NSEW":
        px, py = px + DIRS[d][0] * num, py + DIRS[d][1] * num
    elif d in "LR":
        facing = (facing + ROT[d] * num // 90) % 4
    elif d == "F":
        px, py = px + DIR[facing][0] * num, py + DIR[facing][1] * num
abs(px) + abs(py)
```

```python hidden=true
def rotate_left(x, y, num=1):
    if num == 0:
        return x, y
    else:
        return rotate_left(-y, x, num=num - 1)


def rotate_right(x, y, num=1):
    if num == 0:
        return x, y
    else:
        return rotate_right(y, -x, num=num - 1)
```

```python hidden=true
px, py = 10, 1
tx, ty = 0, 0
for x in l:
    d, num = x[0], int(x[1:])
    if d in "NSEW":
        px, py = px + DIRS[d][0] * num, py + DIRS[d][1] * num
    elif d == "L":
        px, py = rotate_left(px, py, num // 90)
    elif d == "R":
        px, py = rotate_right(px, py, num // 90)
    elif d == "F":
        tx += num * px
        ty += num * py
abs(tx) + abs(ty)
```

<!-- #region hidden=true -->
Let's use complex numbers!
<!-- #endregion -->

```python hidden=true
p = 0
DIRS = "ESWN"
DIR = (1, -1j, -1, 1j)
ROT = dict(zip("LR", (1, -1)))
DIRS = dict(zip(DIRS, DIR))
facing = 1
for x in l:
    d, num = x[0], int(x[1:])
    if d in "NSEW":
        p += DIRS[d] * num
    elif d in "LR":
        facing *= 1j ** (ROT[d] * num // 90)
    elif d == "F":
        p += facing * num
abs(p.real) + abs(p.imag)
```

```python hidden=true
p = 10 + 1j
t = 0
for x in l:
    d, num = x[0], int(x[1:])
    if d in "NSEW":
        p += DIRS[d] * num
    elif d in "LR":
        p *= (1j) ** (ROT[d] * num // 90)
    elif d == "F":
        t += p * num
abs(t.real) + abs(t.imag)
```

<!-- #region heading_collapsed=true -->
## Day 13
<!-- #endregion -->

```python hidden=true
with open("input13.txt") as f:
    l = int(f.readline()[:-1])
    ll = f.read()[:-1].split(",")
ll = [int(x) if x != "x" else None for x in ll]
d = dict(zip(ll, range(0, -len(ll), -1)))
del d[None]
```

<!-- #region hidden=true -->
best = 2 * reduce(lambda a, b: a * b, d.keys())
best_bus = 0
for bus in d.keys():
    new = ((l // bus) + 1) * bus if l % bus != 0 else l
    if new < best:
        best = new
        best_bus = bus
best_bus * (best - l)
<!-- #endregion -->

```python hidden=true
def chinese_remainder(n, a):
    s = 0
    prod = reduce(lambda a, b: a * b, n)
    for n_i, a_i in zip(n, a):
        p = prod // n_i
        s += a_i * mul_inv(p, n_i) * p
    return s % prod


def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1:
        return 1
    while a > 1:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += b0
    return x1
```

```python hidden=true
chinese_remainder(d.keys(), d.values())
```

```python hidden=true
n = 0
step = 1
for bus, i in d.items():
    for c in count(n, step):
        if (c - i) % bus == 0:
            n = c
            break
    step *= bus
n
```

<!-- #region heading_collapsed=true -->
## Day 14
<!-- #endregion -->

```python hidden=true
class Mem(dict):
    def __setitem__(self, key, value):
        global mask
        x = int(mask.replace("0", "1").replace("X", "0"), 2)
        maskb = int(mask.replace("X", "0"), 2)
        value = (((value | x) ^ x)) | maskb
        super().__setitem__(key, value)

mem = Mem()
with open("input14.txt") as f:
    text = f.read()
    l = ["mask" + x for x in text.split("mask")[1:]]
    l = [re.sub("mask = ([01X]+)", r'mask = "\1"', x) for x in l]
for x in l:
    exec(x)
sum(mem.values())
```

```python hidden=true
class Mem2(dict):
    def __setitem__(self, key, value):
        global mask
        X_pos = [i for i, x in enumerate(mask) if x == "X"]
        X_bin = [2 ** (len(mask) - 1 - x) for x in X_pos]
        X_mask = sum(X_bin)
        maskb = int(mask.replace("X", "0"), 2)
        key |= maskb
        key |= X_mask
        key ^= X_mask
        for digits in product(*(range(2) for _ in range(len(X_pos)))):
            super().__setitem__(
                key + sum(x * y for x, y in zip(X_bin, digits)),
                value,
            )
mem = Mem2()
for x in l:
    exec(x)
sum(mem.values())
```

<!-- #region heading_collapsed=true -->
## Day 15
<!-- #endregion -->

```python hidden=true
import numpy as np

from numba import njit
from numba import types
from numba.typed import Dict


nums = np.array([11,0,1,10,5,19], dtype=np.int64)

@njit("int64(int64[:], int64)")
def day15(nums, N):
    last = np.full(N, -1, dtype=np.int64)
    for i, x in enumerate(nums[:-1]):
        last[x] = i
    buffer = nums[-1]
    for i in range(len(nums) - 1, N - 1):
        y = 0 if last[buffer] == -1 else i - last[buffer]
        last[buffer], buffer = i, y
    return buffer

print(day15(nums, 2020))
print(day15(nums, 30000000))
```

<!-- #region heading_collapsed=true -->
## Day 16
<!-- #endregion -->

```python hidden=true
def parse_tickets(filename="input16.txt"):
    global rules, nearby, your
    rules = {}
    with open("input16.txt") as f:
        while True:
            text = f.readline()
            if text.startswith("\n"):
                break
            k, v = text.split(": ")
            v = [tuple(map(int, x.split("-"))) for x in v[:-1].split(" or ")]
            rules[k] = v
        f.readline()  # "your ticket"
        your = list(map(int, f.readline().split(",")[:-1]))
        f.readline()  # blank line
        f.readline()  # "nearby tickets"
        nearby = f.read().splitlines()
        nearby = [tuple(map(int, x.split(","))) for x in nearby]

def is_possible(n, rules):
    return any(x <= n <= y for v in rules.values() for x, y in v)

parse_tickets()
sum(num for ticket in nearby for num in ticket if not is_possible(num, rules))
```

```python hidden=true
nearby_possible = [x for x in nearby if all(is_possible(y, rules) for y in x)]

def check_field(n, rules):
    return any(x <= n <= y for x, y in rules)

possible_fields = [
    {
        k for k, v in rules.items()
        if all(check_field(x[idx], v) for x in nearby_possible)
    }
    for idx in range(20)
]
try_order = sorted(range(20), key=lambda x: len(possible_fields[x]))
sol = [""]  * 20
for idx in try_order:
    if len(possible_fields[idx]) == 1:
        [sol[idx]] = possible_fields[idx]
        for i in range(20):
            possible_fields[i].discard(sol[idx])

math.prod(v for k, v in zip(sol, your) if k.startswith("depart"))
```

## Day 17

```python

```
