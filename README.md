### Install

```shell
# CPU only (default)
pip install nnops

# With MPS acceleration (macOS)
pip install nnops[mps]
```

### Build from source

```shell
# CPU only
python3 scripts/build.py

# With MPS acceleration (macOS)
python3 scripts/build.py --feature mps

# Build wheel without installing
python3 scripts/build.py --wheel
```

### Run tests

```shell
python3 -m pytest tests/
```

### Example
```python
import nnops, nnops.tensor

x = nnops.tensor.randn(2, 3)
y = nnops.tensor.randn(3, 4)

x = (x + 2) * 3
y = (y - 4) / 5
z = x @ y # [2, 4]
```

### Sponsor
<table align="center">
    <thead>
        <tr>
            <th colspan="2">公众号</th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="2"><img src="https://jiauzhang.github.io/ghstatic/images/ofa_m.png" style="height: 196px" alt="AliPay.png"></td>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th>AliPay</th>
            <th>WeChatPay</th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td><img src="https://jiauzhang.github.io/AliPay.png" style="width: 196px; height: 196px" alt="AliPay.png"></td>
            <td><img src="https://jiauzhang.github.io/WeChatPay.png" style="width: 196px; height: 196px" alt="WeChatPay.png"></td>
        </tr>
    </tbody>
</table>
