# SƠ ĐỒ CHO BÀI BÁO NCKH TFT-PPO

Các sơ đồ này sử dụng Mermaid syntax. Có thể render trong:
- VS Code: cài extension "Mermaid Markdown Syntax Highlighting" hoặc "Mermaid Preview"
- GitHub README: tự động render
- Draw.io / Online: import trực tiếp

---

## 1. SYSTEM ARCHITECTURE (Kiến trúc hệ thống)

```mermaid
graph TB
    subgraph Clients["CLIENTS (h9 - h16)"]
        C1[Client 1]
        C2[Client 2]
        C3[Client 3]
        C4[Client 4]
        C5[Client 5]
        C6[Client 6]
        C7[Client 7]
        C8[Client 8]
    end

    subgraph EdgeSwitches["EDGE SWITCHES (s1-s4)"]
        ES1[Switch 1]
        ES2[Switch 2]
        ES3[Switch 3]
        ES4[Switch 4]
    end

    subgraph CoreSwitch["CORE SWITCH (s5)"]
        CS[OpenFlow 1.3]
    end

    subgraph Controller["RYU SDN CONTROLLER"]
        subgraph TFT_PPO["TFT-PPO Agent"]
            subgraph Encoder["TFT Encoder"]
                LSTM[LSTM Encoder]
                MHA[Multi-Head Attention]
                FL[Feature Linear]
            end
            
            subgraph ActorCritic["Actor-Critic"]
                Actor[Actor Head<br/>π(a|s) - Server 0-2]
                Critic[Critic Head<br/>V(s) - Value Estimate]
            end
        end
        
        subgraph Safety["Safety Override"]
            Check{utilization<br/> > 0.95?}
            WRR[WRR Fallback]
        end
    end

    subgraph Backend["BACKEND SERVERS"]
        S1[Server h5<br/>10 Mbps]
        S2[Server h7<br/>50 Mbps]
        S3[Server h8<br/>100 Mbps]
    end

    C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 --> ES1 & ES2 & ES3 & ES4
    ES1 & ES2 & ES3 & ES4 --> CS
    CS --> Controller
    LSTM <--> MHA
    MHA --> FL
    FL --> Actor
    FL --> Critic
    Actor --> Check
    Check -->|Yes| WRR
    Check -->|No| Actor
    WRR --> S1 & S2 & S3
    Actor --> S1 & S2 & S3
    
    style Clients fill:#e1f5fe
    style Controller fill:#fff3e0
    style Backend fill:#e8f5e9
    style TFT_PPO fill:#f3e5f5
    style Safety fill:#ffebee
```

---

## 2. TFT ENCODER (Bộ mã hóa TFT)

```mermaid
graph LR
    subgraph Input["Input: 20 Features"]
        F1[Load h5]
        F2[Load h7]
        F3[Load h8]
        F4[Latency]
        F5[Queue Length]
        F6[Cache Hit]
        F7[Traffic Intensity]
        F8[... other 13]
    end

    subgraph Temporal["Temporal Processing"]
        LSTM1[LSTM Cell 1]
        LSTM2[LSTM Cell 2]
        LSTM3[LSTM Cell T]
    end

    subgraph Attention["Multi-Head Attention"]
        H1[Head 1]
        H2[Head 2]
        H3[Head h]
    end

    subgraph Output["Output"]
        ActorOut[Actor Head]
        CriticOut[Critic Head]
    end

    F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 --> LSTM1
    LSTM1 --> LSTM2
    LSTM2 --> LSTM3
    LSTM3 --> H1 & H2 & H3
    H1 & H2 & H3 --> ActorOut
    H1 & H2 & H3 --> CriticOut

    style Input fill:#e1f5fe
    style Temporal fill:#fff3e0
    style Attention fill:#f3e5f5
    style Output fill:#e8f5e9
```

---

## 3. PPO ALGORITHM FLOW

```mermaid
flowchart TD
    Start([Start Training]) --> Init[Initialize Policy πθ<br/>Load 20-dim state]
    
    Init --> Episode{For each Episode}
    
    Episode -->|200 steps| Collect[Collect Trajectories]
    Collect --> Action[Agent selects action<br/>a ∈ {0, 1, 2}]
    Action --> Env[Environment responds]
    Env --> Reward[Calculate reward<br/>r = balance + throughput<br/>- latency - overload]
    
    Reward --> Check{Step < 200?}
    Check -->|Yes| Store[Store (s, a, r) tuple]
    Store --> Action
    
    Check -->|No| GAE[Compute Advantage Â<br/>using GAE λ=0.95]
    
    GAE --> PPO[PPO Update:<br/>LCLIP = min(r̂Â, clip(r̂)Â)]
    
    PPO --> Clip{clip ratio<br/>in [0.8, 1.2]?}
    Clip -->|Yes| Update[Update θ with<br/>Adam optimizer]
    Clip -->|No| Clamp[Clamp update]
    
    Update & Clamp --> Loss[Compute loss:<br/>L = LCLIP + c1LVF + c2S]
    
    Loss --> CheckLoss{Loss<br/>converging?}
    CheckLoss -->|No| Episode
    CheckLoss -->|Yes| Save[Save checkpoint<br/>every 50K steps]
    
    Save --> CheckSteps{Steps <<br/>500K?}
    CheckSteps -->|Yes| Episode
    CheckSteps -->|No| Done([Training Done])
```

---

## 4. SAFETY OVERRIDE MECHANISM

```mermaid
stateDiagram-v2
    [*] --> Normal: Start
    Normal --> Monitoring: Every inference
    Monitoring --> CheckUtil: Check utilization
    CheckUtil -->|u ≤ 0.95| Normal: Safe
    CheckUtil -->|u > 0.95| Override: Threshold exceeded
    Override --> WRRLB: Bypass Agent
    WRRLB --> Monitoring: Next request
    Normal --> PPOAgent: Agent decision
    PPOAgent --> Monitoring: Apply action
```

---

## 5. REINFORCEMENT LEARNING CYCLE

```mermaid
sequenceDiagram
    participant Env as SDN Environment
    participant Agent as TFT-PPO Agent
    participant Stats as PortStats Collector
    
    Note over Env,Agent: Episode Start
    
    loop Every Request
        Env->>Stats: Collect metrics
        Stats-->>Agent: State s_t (20 features)
        
        Agent->>Agent: LSTM encode temporal
        Agent->>Agent: Attention weights
        Agent->>Agent: π(a|s), V(s)
        
        Agent-->>Env: Action a ∈ {0,1,2}
        
        Env->>Env: Apply action
        Env->>Stats: Measure result
        
        Stats-->>Agent: Reward r_t
        Agent->>Agent: Store transition
        
        alt Every 2048 steps
            Agent->>Agent: PPO Update
        end
    end
```

---

## 6. TRAINING PIPELINE

```mermaid
graph TD
    subgraph DataCollection["Data Collection"]
        Mininet[Mininet Emulator]
        Ryu[Ryu Controller]
        Stats[OpenFlow Stats]
    end

    subgraph Simulation["Gymnasium Simulation"]
        Env[SDNEnvV3Real]
        InitCap[Random Capacity<br/>±20% per episode]
    end

    subgraph PPO_Training["PPO Training"]
        Collect[Collect 2048 steps]
        GAE[Compute GAE]
        Clip[PPO Clipped Objective]
        Adam[Adam Optimizer]
    end

    subgraph Evaluation["Evaluation"]
        WRR[WRR Baseline]
        PPO[PPO Agent]
        Compare[Compare metrics]
    end

    Mininet --> Env
    Ryu --> Stats
    Stats --> Env
    InitCap --> Env
    
    Env --> Collect
    Collect --> GAE
    GAE --> Clip
    Clip --> Adam
    
    Env --> WRR
    Env --> PPO
    WRR --> Compare
    PPO --> Compare
    Compare --> Report[Final Report]

    style DataCollection fill:#e1f5fe
    style Simulation fill:#fff3e0
    style PPO_Training fill:#f3e5f5
    style Evaluation fill:#e8f5e9
```

---

## 7. NETWORK TOPOLOGY (Fat-Tree K=4)

```mermaid
graph TB
    subgraph Layer2["Edge Layer"]
        E1(("s1<br/>Edge"))
        E2(("s2<br/>Edge"))
        E3(("s3<br/>Edge"))
        E4(("s4<br/>Edge"))
    end

    subgraph Layer1["Core Layer"]
        C1(("s5<br/>Core"))
    end

    subgraph Servers["Backend Servers"]
        S1["h5: 10 Mbps"]
        S2["h7: 50 Mbps"]
        S3["h8: 100 Mbps"]
    end

    subgraph Clients["Clients"]
        CL1["h9"]
        CL2["h10"]
        CL3["h11"]
        CL4["h12"]
        CL5["h13"]
        CL6["h14"]
        CL7["h15"]
        CL8["h16"]
    end

    CL1 & CL2 --> E1
    CL3 & CL4 --> E2
    CL5 & CL6 --> E3
    CL7 & CL8 --> E4

    E1 & E2 & E3 & E4 --> C1
    C1 --> S1 & S2 & S3

    style Layer2 fill:#bbdefb
    style Layer1 fill:#90caf9
    style Servers fill:#c8e6c9
    style Clients fill:#ffccbc
```

---

## 8. HARDWARE DEGRADATION SCENARIO

```mermaid
sequenceDiagram
    participant Client
    participant Switch
    participant Controller
    participant Server1 as "h5 (10M)"
    participant Server2 as "h7 (50M)"
    participant Server3 as "h8 (100M)"

    Note over Server3: Hardware Degradation!<br/>Capacity drops to 50M

    Client->>Switch: HTTP Request
    Switch->>Controller: Packet-in
    Controller->>Controller: Check utilization

    Note over Controller: h8 utilization > 95%<br/>PPO detects anomaly!

    Controller->>Server2: Route to h7 (50M)
    Server2-->>Switch: Response
    Switch-->>Client: Response

    Note over Controller: PPO learned to avoid<br/>degraded server!

    loop Every Request
        Client->>Switch: Request
        Switch->>Controller: Packet-in
        Controller->>Controller: PPO decision
        Controller->>Server2: Route to healthy server
        Server2-->>Switch: Response
        Switch-->>Client: Response
    end
```

---

## 9. REWARD FUNCTION DECOMPOSITION

```mermaid
graph LR
    subgraph Components["Reward Components"]
        Balance["balance_bonus<br/>= balance_score × 3.0"]
        Throughput["throughput_bonus<br/>based on achieved BW"]
        Latency["latency_penalty<br/>∝ avg latency"]
        Overload["overload_penalty<br/>= 20000 if u > 0.95"]
    end

    subgraph Calculation["Reward Calculation"]
        Sum["reward = b + t - l - o"]
    end

    Balance --> Sum
    Throughput --> Sum
    Latency --> Sum
    Overload --> Sum

    Sum --> Agent[PPO Agent]
    Agent --> Decision[Server Selection<br/>a ∈ {0, 1, 2}]

    style Components fill:#fff3e0
    style Calculation fill:#e8f5e9
    style Agent fill:#f3e5f5
```

---

## 10. COMPARISON: WRR vs PPO APPROACH

```mermaid
graph LR
    subgraph WRR["WRR (Static)"]
        W1[Round Robin]
        W2[Fixed Weights]
        W3[No Adaptation]
    end

    subgraph PPO["PPO (Adaptive)"]
        P1[Learn Patterns]
        P2[Dynamic Weights]
        P3[Adapt to Drift]
    end

    W1 & W2 & W3 --> WRR_Res[Thất bại khi<br/>hardware degradation]
    P1 & P2 & P3 --> PPO_Res[Thành công +8.6%<br/>trong degradation]

    WRR_Res -.->|Nhưng| NormalRes[Tốt trong<br/>điều kiện bình thường]
    PPO_Res -.->|Nhưng| NormalRes

    style WRR fill:#ffcdd2
    style PPO fill:#c8e6c9
    style NormalRes fill:#fff9c4
```

---

## Hướng dẫn sử dụng:

### VS Code (Live Preview):
1. Cài extension "Markdown Preview Mermaid Support"
2. Mở file .md và click "Open Preview"
3. Các sơ đồ sẽ tự động render

### Export sang PNG/SVG:
1. Copy Mermaid code vào https://mermaid.live
2. Click "Export" → PNG/SVG

### Draw.io:
1. Mở https://app.diagrams.net
2. File → Import → Mermaid
3. Chỉnh sửa và export
