import torch
import torch.nn as nn
from torch.distributions import Normal


# π(a|s)
# 入力が観測値、出力が行動確率分布
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, battery_capacity, hidden_dims=[256,256], log_std_bounds=(-20,2)):
        super().__init__()
        self.battery_capacity = battery_capacity
        self.log_std_min, self.log_std_max = log_std_bounds
        self.eps = 1e-6
        layers, last_dim = [], obs_dim

        for dim in hidden_dims:
            layers += [nn.Linear(last_dim, dim), nn.ReLU()]
            last_dim = dim
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(last_dim, act_dim)
        self.log_std = nn.Linear(last_dim, act_dim)

    def _dist(self, obs):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return Normal(mean, std)

    def forward(self, obs):
        d = self._dist(obs)
        return d.mean, d.stddev
    
    def sample(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist = self._dist(obs)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)

        pv  = obs[:, 0:1]
        soc = obs[:, 3:4]  
        ## ---------------------------------------------
        # 充電上限値の計算, minimum(PVout, SoC空き容量)
        remaining_soc = (1 - soc) * self.battery_capacity
        pv_charge_limit = pv * self.battery_capacity * 0.5
        max_charge = torch.minimum(pv_charge_limit, remaining_soc)
        # 放電上限値の計算, SoCの下限値を下回らない放電量
        max_discharge = torch.minimum(soc*self.battery_capacity, torch.full_like(soc, self.battery_capacity*0.5))
        ## ---------------------------------------------
        scale = torch.where(y_t < 0, max_charge, max_discharge)

        action = y_t * scale

        log_prob = dist.log_prob(x_t).sum(-1, keepdim=True)
        log_prob -= torch.log(scale * (1 - y_t.pow(2)) + self.eps).sum(-1, keepdim=True)
        return action.squeeze(0), log_prob.squeeze(0)
    
    @torch.no_grad()
    def act(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mean = self._dist(obs).mean
        yt_mean = torch.tanh(mean)
        # 同じ動的スケールを適用
        pv  = obs[:, 0:1]
        soc = obs[:, 3:4]  
         # 充電上限値の計算, minimum(PVout, SoC空き容量)
        remaining_soc = (1 - soc) * self.battery_capacity
        pv_charge_limit = pv * self.battery_capacity * 0.5
        max_charge = torch.minimum(pv_charge_limit, remaining_soc)
        # 放電上限値の計算, SoCの下限値を下回らない放電量
        max_discharge = torch.minimum(soc*self.battery_capacity, torch.full_like(soc, self.battery_capacity*0.5))
        ## ---------------------------------------------
        scale = torch.where(yt_mean < 0, max_charge, max_discharge)
        return yt_mean * scale

# Q(s,a)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=[256,256]):
        super().__init__()
        layers = []
        last_dim = obs_dim + act_dim
        for dim in hidden_dims:
            layers += [nn.Linear(last_dim, dim), nn.ReLU()]
            last_dim = dim
        layers.append(nn.Linear(last_dim, 1)) # 状態行動の価値を出力
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
    
# デコーダ
class FeatureDecoder(nn.Module):
    def __init__(self, obs_dim, act_dim, feature_dim, hidden_dims=[256,256]):
        super().__init__()
        layers = []
        last_dim = feature_dim
        for dim in hidden_dims:
            layers += [nn.Linear(last_dim, dim), nn.ReLU()]
            last_dim = dim
        layers += [nn.Linear(last_dim, obs_dim + act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

# Encder
# (s,a) を入力して φ_ζ ∈ R^feature_dim を出す
class FeatureEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, feature_dim, hidden_dims=[256,256]):
        super().__init__()
        layers = []
        last_dim = obs_dim + act_dim
        for dim in hidden_dims:
            layers += [nn.Linear(last_dim, dim), nn.ReLU()]
            last_dim = dim
        layers += [nn.Linear(last_dim, feature_dim)]
        self.net = nn.Sequential(*layers)
        self.output_dim = feature_dim

    def forward(self, s, a):
        # s: (obs_dim,), a: (act_dim,)
        x = torch.cat([s, a], dim=-1)   # → (obs_dim+act_dim,)
        return self.net(x)              # → (feature_dim,)