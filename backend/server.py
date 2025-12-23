# backend/server.py
from bottle import route, run, request, static_file, response, hook
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import os

@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

# Уравнения
EQUATIONS = {
    "eq1": {
        "name": "dy/dx + (x + lq(x))·y = x³ + 2x + lq(x)·x², y(0)=1",
        "y0": 1.0,
        "x_range": (0, 2),
        "lq": lambda x: (1 + 3 * x**2) / (1 + x + x**3),
        "rhs": lambda x, y: x**3 + 2*x + ((1 + 3*x**2)/(1 + x + x**3))*x**2 - (x + (1 + 3*x**2)/(1 + x + x**3))*y,
        "residual": lambda x, y, dy_dx, lq_func: dy_dx + (x + lq_func(x))*y - x**3 - 2*x - lq_func(x)*x**2
    },
    "eq2": {
        "name": "dy/dx = -2*x*y, y(0)=1",
        "y0": 1.0,
        "x_range": (-2, 2),
        "rhs": lambda x, y: -2 * x * y,
        "residual": lambda x, y, dy_dx, lq_func: dy_dx + 2 * x * y,
        "lq": lambda x: 0
    },
    "eq3": {
        "name": "dy/dx = y*(1 - y), y(0)=0.1",
        "y0": 0.1,
        "x_range": (0, 5),
        "rhs": lambda x, y: y * (1 - y),
        "residual": lambda x, y, dy_dx, lq_func: dy_dx - y * (1 - y),
        "lq": lambda x: 0
    }
}

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 1)
        self.tanh = nn.Tanh()
        # Инициализация весов
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        h = self.tanh(self.fc1(x))
        return self.fc2(h)

@route('/solve', method=['OPTIONS', 'POST'])
def solve_ode():
    data = request.json
    eq_key = data.get("equation", "eq1")
    if eq_key not in EQUATIONS:
        return {"error": f"Неизвестное уравнение: {eq_key}"}

    eq = EQUATIONS[eq_key]
    x_start, x_end = eq["x_range"]
    x_train = np.linspace(x_start, x_end, 60)
    y0 = eq["y0"]

    # Численное решение
    try:
        sol = solve_ivp(eq["rhs"], [x_start, x_end], [y0], t_eval=x_train, method='RK45')
        if not sol.success:
            return {"error": "Численное решение не сошлось"}
        y_num = sol.y[0]
    except Exception as e:
        return {"error": f"Ошибка численного решения: {str(e)}"}

    # Нейросетевое решение
    model = Net()
    x_t = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
    y0_t = torch.tensor(y0, dtype=torch.float32)

    def compute_loss(model, x):
        x = x.clone().detach().requires_grad_(True)
        y = model(x)
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        # Передаём lq_func в residual
        lq_func = eq.get("lq", lambda x: 0)
        residual = eq["residual"](x, y, dy_dx, lq_func)
        loss_pde = (residual ** 2).mean()
        idx0 = np.argmin(np.abs(x_train - x_start))
        loss_bc = (y[idx0] - y0_t) ** 2
        return loss_pde + loss_bc

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(5000):
        optimizer.zero_grad()
        loss = compute_loss(model, x_t)
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-6:
            break

    with torch.no_grad():
        y_pred = model(x_t).squeeze().numpy()

    return {
        "x": [round(float(x), 6) for x in x_train.tolist()],
        "numerical": [round(float(y), 6) for y in y_num.tolist()],
        "neural": [round(float(y), 6) for y in y_pred.tolist()]
    }

@route('/')
def index():
    return static_file('index.html', root='../frontend')

@route('/<filepath:path>')
def static(filepath):
    return static_file(filepath, root='../frontend')

if __name__ == '__main__':
    print("🚀 Сервер запущен на http://127.0.0.1:8080")
    run(host='127.0.0.1', port=8080, debug=True)