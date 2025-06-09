# A2A Marketplace Simulator - Complete Implementation

## Project Structure
```
a2a_marketplace/
├── README.md
├── requirements.txt
├── main.py
├── agents/
│   ├── __init__.py
│   ├── agent_base.py
│   ├── agent_a.py
│   ├── agent_b.py
│   └── agent_c.py
├── communication/
│   ├── __init__.py
│   ├── message_schema.py
│   └── p2p_discovery.py
├── ledger/
│   ├── __init__.py
│   ├── mock_ledger.py
│   └── token_contract.sol
├── ai/
│   ├── __init__.py
│   ├── rl_negotiation.py
│   ├── bandit_selection.py
│   └── service_matcher.py
├── storage/
│   ├── __init__.py
│   └── storage_interface.py
├── blockchain/
│   ├── __init__.py
│   ├── token_contract.sol
│   └── voting_contract.sol
└── ui/
    ├── __init__.py
    ├── streamlit_ui.py
    └── interaction_ui.py
```

## File Implementations

### 1. README.md
```markdown
# A2A Marketplace Simulator

A simulation environment for autonomous agent-to-agent economic interactions featuring:
- Service discovery and negotiation
- AI-powered price negotiation using Reinforcement Learning
- Multi-armed bandit agent selection
- Token-based payment system
- Interactive web dashboard

## Features
- **Agent Services**: Data cleaning, text translation, document summarization
- **Smart Negotiation**: RL-based price negotiation
- **Intelligent Selection**: Bandit algorithms for optimal agent selection
- **Payment System**: Mock token-based transactions
- **Web Interface**: Streamlit dashboard for interaction

## Quick Start
```bash
pip install -r requirements.txt
python main.py  # CLI simulation
streamlit run ui/interaction_ui.py  # Web interface
```

## Architecture
- **Agents**: Autonomous service providers with AI decision-making
- **Communication**: P2P discovery and message passing
- **AI Layer**: RL negotiation + bandit selection algorithms
- **Ledger**: Token-based payment system
- **UI**: Interactive simulation dashboard
```

### 2. requirements.txt
```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
web3>=6.0.0
ipfshttpclient>=0.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 3. agents/__init__.py
```python
# Empty file to make agents a package
```

### 4. agents/agent_base.py
```python
from communication.message_schema import ServiceRequest, ServiceResponse
from ledger.mock_ledger import MockLedger
import random
import time

class AgentBase:
    def __init__(self, name, services, ledger: MockLedger):
        self.name = name
        self.services = services  # {service_name: base_price}
        self.ledger = ledger
        self.balance = 100
        self.service_history = []
        self.success_rate = 0.95
        self.response_time = random.uniform(1, 3)  # seconds
        self.load = 0  # current workload
        self.reputation = 5.0  # out of 5
        
    def list_services(self):
        """Return available services with current pricing"""
        return {
            service: {
                'price': price,
                'success_rate': self.success_rate,
                'response_time': self.response_time,
                'reputation': self.reputation
            }
            for service, price in self.services.items()
        }
    
    def get_dynamic_price(self, service_type, base_demand=1.0):
        """Calculate dynamic pricing based on load and demand"""
        base_price = self.services.get(service_type, 0)
        load_multiplier = 1 + (self.load * 0.1)  # 10% increase per load unit
        demand_multiplier = base_demand
        return int(base_price * load_multiplier * demand_multiplier)
    
    def handle_request(self, request: ServiceRequest):
        """Process incoming service request"""
        if request.service_type not in self.services:
            return ServiceResponse(
                success=False,
                message="Service not offered",
                cost=0,
                execution_time=0
            )
        
        # Calculate actual price
        actual_price = self.get_dynamic_price(request.service_type)
        
        # Simulate negotiation acceptance (could be enhanced with RL)
        if request.offered_price >= actual_price * 0.8:  # Accept if >= 80% of asking price
            # Process payment
            if self.ledger.transfer(request.sender, self.name, request.offered_price):
                # Simulate service execution
                execution_time = random.uniform(1, 5)
                time.sleep(0.1)  # Simulate processing
                
                # Update history
                self.service_history.append({
                    'service': request.service_type,
                    'price': request.offered_price,
                    'client': request.sender,
                    'timestamp': time.time()
                })
                
                self.load = max(0, self.load - 1)  # Reduce load after completion
                
                return ServiceResponse(
                    success=True,
                    message=f"{self.name} completed {request.service_type} for {request.sender}",
                    cost=request.offered_price,
                    execution_time=execution_time
                )
            else:
                return ServiceResponse(
                    success=False,
                    message="Payment failed - insufficient funds",
                    cost=0,
                    execution_time=0
                )
        else:
            return ServiceResponse(
                success=False,
                message=f"Price too low. Minimum: {actual_price}",
                cost=actual_price,
                execution_time=0
            )
    
    def update_reputation(self, rating):
        """Update agent reputation based on client feedback"""
        self.reputation = (self.reputation * 0.9) + (rating * 0.1)
        self.reputation = max(1.0, min(5.0, self.reputation))
```

### 5. agents/agent_a.py
```python
from agents.agent_base import AgentBase
from ledger.mock_ledger import ledger_instance

# Data Processing Specialist
agent_a = AgentBase(
    name="DataProcessor_A", 
    services={
        "clean_data": 5,
        "validate_data": 3,
        "transform_data": 7
    }, 
    ledger=ledger_instance
)

# Enhance with specialization
agent_a.success_rate = 0.98
agent_a.response_time = 1.5
agent_a.reputation = 4.8
```

### 6. agents/agent_b.py
```python
from agents.agent_base import AgentBase
from ledger.mock_ledger import ledger_instance

# Language Services Specialist
agent_b = AgentBase(
    name="Translator_B", 
    services={
        "translate_text": 4,
        "summarize_text": 6,
        "analyze_sentiment": 5
    }, 
    ledger=ledger_instance
)

# Enhance with specialization
agent_b.success_rate = 0.95
agent_b.response_time = 2.0
agent_b.reputation = 4.6
```

### 7. agents/agent_c.py
```python
from agents.agent_base import AgentBase
from ledger.mock_ledger import ledger_instance

# Computation Services Specialist
agent_c = AgentBase(
    name="Computer_C", 
    services={
        "run_analysis": 8,
        "generate_report": 10,
        "optimize_model": 15
    }, 
    ledger=ledger_instance
)

# Enhance with specialization
agent_c.success_rate = 0.92
agent_c.response_time = 3.5
agent_c.reputation = 4.4
```

### 8. communication/__init__.py
```python
# Empty file to make communication a package
```

### 9. communication/message_schema.py
```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class ServiceRequest:
    sender: str
    receiver: str
    service_type: str
    offered_price: int
    deadline: Optional[float] = None
    requirements: Optional[dict] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ServiceResponse:
    success: bool
    message: str
    cost: int
    execution_time: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class NegotiationOffer:
    sender: str
    receiver: str
    service_type: str
    proposed_price: int
    counter_offer: bool = False
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
```

### 10. communication/p2p_discovery.py
```python
from typing import Dict, List, Optional
import random

class P2PNetwork:
    def __init__(self):
        self.registry = {}  # {agent_name: {services, metadata}}
        self.network_graph = {}  # Simple peer connections
        
    def register(self, agent_name: str, services: dict, metadata: dict = None):
        """Register an agent with its services"""
        self.registry[agent_name] = {
            'services': services,
            'metadata': metadata or {},
            'online': True,
            'last_seen': None
        }
        
    def discover(self, service_type: str) -> List[dict]:
        """Find all agents offering a specific service"""
        providers = []
        for agent_name, info in self.registry.items():
            if info['online'] and service_type in info['services']:
                providers.append({
                    'agent': agent_name,
                    'price': info['services'][service_type]['price'],
                    'reputation': info['services'][service_type].get('reputation', 3.0),
                    'success_rate': info['services'][service_type].get('success_rate', 0.9),
                    'response_time': info['services'][service_type].get('response_time', 2.0)
                })
        return sorted(providers, key=lambda x: x['reputation'], reverse=True)
    
    def find_best_provider(self, service_type: str, criteria: str = 'reputation') -> Optional[str]:
        """Find the best provider based on specified criteria"""
        providers = self.discover(service_type)
        if not providers:
            return None
            
        if criteria == 'reputation':
            return max(providers, key=lambda x: x['reputation'])['agent']
        elif criteria == 'price':
            return min(providers, key=lambda x: x['price'])['agent']
        elif criteria == 'speed':
            return min(providers, key=lambda x: x['response_time'])['agent']
        else:
            return random.choice(providers)['agent']
    
    def get_network_stats(self):
        """Get network statistics"""
        total_agents = len(self.registry)
        online_agents = sum(1 for info in self.registry.values() if info['online'])
        all_services = set()
        for info in self.registry.values():
            all_services.update(info['services'].keys())
            
        return {
            'total_agents': total_agents,
            'online_agents': online_agents,
            'unique_services': len(all_services),
            'services': list(all_services)
        }

# Global network instance
p2p_network = P2PNetwork()
```

### 11. ledger/__init__.py
```python
# Empty file to make ledger a package
```

### 12. ledger/mock_ledger.py
```python
from typing import Dict, List, Optional
import time
import json

class Transaction:
    def __init__(self, sender: str, receiver: str, amount: int, service: str = None):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.service = service
        self.timestamp = time.time()
        self.tx_id = f"tx_{int(self.timestamp)}_{hash(f'{sender}{receiver}{amount}') % 10000}"

class MockLedger:
    def __init__(self):
        self.accounts = {}
        self.transaction_history = []
        self.locked_funds = {}  # For escrow-like functionality
        
    def create_account(self, agent_name: str, initial_balance: int = 100):
        """Create a new account with initial balance"""
        self.accounts[agent_name] = initial_balance
        
    def get_balance(self, agent_name: str) -> int:
        """Get account balance"""
        return self.accounts.get(agent_name, 0)
    
    def transfer(self, sender: str, receiver: str, amount: int, service: str = None) -> bool:
        """Transfer tokens between accounts"""
        if self.accounts.get(sender, 0) >= amount:
            self.accounts[sender] -= amount
            self.accounts[receiver] = self.accounts.get(receiver, 0) + amount
            
            # Record transaction
            tx = Transaction(sender, receiver, amount, service)
            self.transaction_history.append(tx)
            
            return True
        return False
    
    def lock_funds(self, agent_name: str, amount: int, purpose: str) -> str:
        """Lock funds for escrow (e.g., during negotiation)"""
        if self.accounts.get(agent_name, 0) >= amount:
            self.accounts[agent_name] -= amount
            lock_id = f"lock_{int(time.time())}_{hash(purpose) % 10000}"
            self.locked_funds[lock_id] = {
                'agent': agent_name,
                'amount': amount,
                'purpose': purpose,
                'timestamp': time.time()
            }
            return lock_id
        return None
    
    def release_funds(self, lock_id: str, to_agent: str) -> bool:
        """Release locked funds to specified agent"""
        if lock_id in self.locked_funds:
            locked = self.locked_funds.pop(lock_id)
            self.accounts[to_agent] = self.accounts.get(to_agent, 0) + locked['amount']
            return True
        return False
    
    def return_locked_funds(self, lock_id: str) -> bool:
        """Return locked funds to original owner"""
        if lock_id in self.locked_funds:
            locked = self.locked_funds.pop(lock_id)
            self.accounts[locked['agent']] += locked['amount']
            return True
        return False
    
    def get_transaction_history(self, agent_name: str = None) -> List[Transaction]:
        """Get transaction history for an agent or all transactions"""
        if agent_name:
            return [tx for tx in self.transaction_history 
                   if tx.sender == agent_name or tx.receiver == agent_name]
        return self.transaction_history
    
    def get_ledger_stats(self):
        """Get ledger statistics"""
        total_supply = sum(self.accounts.values()) + sum(
            lock['amount'] for lock in self.locked_funds.values()
        )
        return {
            'total_accounts': len(self.accounts),
            'total_supply': total_supply,
            'total_transactions': len(self.transaction_history),
            'locked_funds': len(self.locked_funds)
        }

# Global ledger instance
ledger_instance = MockLedger()

# Initialize default accounts
ledger_instance.create_account("DataProcessor_A", 100)
ledger_instance.create_account("Translator_B", 100)
ledger_instance.create_account("Computer_C", 100)
ledger_instance.create_account("Client_X", 200)  # Test client
```

### 13. ai/__init__.py
```python
# Empty file to make ai a package
```

### 14. ai/rl_negotiation.py
```python
import numpy as np
import random
from typing import Dict, Tuple, Optional

class RLNegotiator:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {}  # {state: {action: q_value}}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # exploration rate
        self.actions = ["accept", "reject", "counter_low", "counter_high"]
        self.negotiation_history = []
        
    def get_state(self, service_type: str, offered_price: int, market_price: int, 
                  agent_reputation: float, urgency: float = 0.5) -> str:
        """Convert negotiation context to state string"""
        price_ratio = offered_price / max(market_price, 1)
        price_category = "low" if price_ratio < 0.8 else "fair" if price_ratio < 1.2 else "high"
        reputation_category = "low" if agent_reputation < 3.0 else "medium" if agent_reputation < 4.5 else "high"
        urgency_category = "low" if urgency < 0.3 else "medium" if urgency < 0.7 else "high"
        
        return f"{service_type}_{price_category}_{reputation_category}_{urgency_category}"
    
    def get_action(self, state: str, explore: bool = True) -> str:
        """Get action based on current state using epsilon-greedy policy"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Choose action with highest Q-value
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str = None):
        """Update Q-value using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        if next_state and next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if next_state else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def negotiate(self, service_type: str, initial_offer: int, market_price: int,
                  agent_reputation: float, max_rounds: int = 3) -> Tuple[str, int]:
        """Conduct a negotiation session"""
        current_offer = initial_offer
        for round_num in range(max_rounds):
            state = self.get_state(service_type, current_offer, market_price, 
                                 agent_reputation, urgency=round_num/max_rounds)
            
            action = self.get_action(state)
            
            if action == "accept":
                reward = self.calculate_reward(current_offer, market_price, True)
                self.update_q_value(state, action, reward)
                return "accept", current_offer
            
            elif action == "reject":
                reward = self.calculate_reward(current_offer, market_price, False)
                self.update_q_value(state, action, reward)
                return "reject", 0
            
            elif action == "counter_low":
                current_offer = max(1, int(current_offer * 0.8))
                reward = -0.1  # Small penalty for prolonging negotiation
                self.update_q_value(state, action, reward)
                
            elif action == "counter_high":
                current_offer = int(current_offer * 1.2)
                reward = -0.1
                self.update_q_value(state, action, reward)
        
        # If max rounds reached, accept current offer
        final_state = self.get_state(service_type, current_offer, market_price, 
                                   agent_reputation, urgency=1.0)
        reward = self.calculate_reward(current_offer, market_price, True)
        self.update_q_value(final_state, "accept", reward)
        
        return "accept", current_offer
    
    def calculate_reward(self, final_price: int, market_price: int, deal_made: bool) -> float:
        """Calculate reward for negotiation outcome"""
        if not deal_made:
            return -1.0  # Penalty for failed negotiation
        
        # Reward based on how good the deal is compared to market price
        price_ratio = final_price / market_price
        if price_ratio > 1.2:
            return 1.0  # Great deal
        elif price_ratio > 1.0:
            return 0.5  # Good deal
        elif price_ratio > 0.8:
            return 0.0  # Fair deal
        else:
            return -0.5  # Poor deal
    
    def get_strategy_stats(self):
        """Get statistics about learned strategies"""
        if not self.q_table:
            return {"message": "No learning data available"}
        
        action_preferences = {action: 0 for action in self.actions}
        for state_actions in self.q_table.values():
            best_action = max(state_actions, key=state_actions.get)
            action_preferences[best_action] += 1
        
        return {
            "states_learned": len(self.q_table),
            "action_preferences": action_preferences,
            "exploration_rate": self.epsilon
        }

# Global negotiator instance
global_negotiator = RLNegotiator()
```

### 15. ai/bandit_selection.py
```python
import numpy as np
import random
from typing import List, Dict, Any
import time

class MultiArmedBandit:
    def __init__(self, agents: List[str], initial_confidence: float = 1.0):
        self.agents = agents
        self.rewards = {agent: [] for agent in agents}
        self.selection_count = {agent: 0 for agent in agents}
        self.cumulative_reward = {agent: 0.0 for agent in agents}
        self.confidence_interval = {agent: initial_confidence for agent in agents}
        self.total_selections = 0
        
    def select_epsilon_greedy(self, epsilon: float = 0.1) -> str:
        """Epsilon-greedy selection strategy"""
        if random.random() < epsilon or self.total_selections == 0:
            # Explore: random selection
            return random.choice(self.agents)
        else:
            # Exploit: choose agent with highest average reward
            avg_rewards = {
                agent: self.cumulative_reward[agent] / max(self.selection_count[agent], 1)
                for agent in self.agents
            }
            return max(avg_rewards, key=avg_rewards.get)
    
    def select_ucb1(self, c: float = 2.0) -> str:
        """Upper Confidence Bound selection strategy"""
        if self.total_selections == 0:
            return random.choice(self.agents)
        
        ucb_values = {}
        for agent in self.agents:
            if self.selection_count[agent] == 0:
                ucb_values[agent] = float('inf')  # Select unplayed agents first
            else:
                avg_reward = self.cumulative_reward[agent] / self.selection_count[agent]
                confidence = c * np.sqrt(np.log(self.total_selections) / self.selection_count[agent])
                ucb_values[agent] = avg_reward + confidence
        
        return max(ucb_values, key=ucb_values.get)
    
    def select_thompson_sampling(self) -> str:
        """Thompson Sampling selection strategy"""
        samples = {}
        for agent in self.agents:
            # Use Beta distribution for reward sampling
            successes = max(1, sum(1 for r in self.rewards[agent] if r > 0.5))
            failures = max(1, len(self.rewards[agent]) - successes + 1)
            samples[agent] = np.random.beta(successes, failures)
        
        return max(samples, key=samples.get)
    
    def update_reward(self, agent: str, reward: float):
        """Update reward for selected agent"""
        self.rewards[agent].append(reward)
        self.cumulative_reward[agent] += reward
        self.selection_count[agent] += 1
        self.total_selections += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics"""
        stats = {}
        for agent in self.agents:
            count = self.selection_count[agent]
            if count > 0:
                avg_reward = self.cumulative_reward[agent] / count
                recent_rewards = self.rewards[agent][-10:]  # Last 10 rewards
                recent_avg = np.mean(recent_rewards) if recent_rewards else 0
            else:
                avg_reward = 0
                recent_avg = 0
            
            stats[agent] = {
                'selections': count,
                'avg_reward': round(avg_reward, 3),
                'recent_avg': round(recent_avg, 3),
                'total_reward': round(self.cumulative_reward[agent], 2)
            }
        
        return stats

class ContextualBandit:
    def __init__(self, agents: List[str], context_dimensions: int = 4):
        self.agents = agents
        self.context_dim = context_dimensions
        # Simple linear model: reward = context * weights
        self.weights = {agent: np.random.normal(0, 0.1, context_dimensions) 
                      for agent in agents}
        self.history = []
        
    def get_context(self, service_type: str, urgency: float, budget: int, 
                   time_of_day: float) -> np.ndarray:
        """Convert situational factors to context vector"""
        service_complexity = {
            'clean_data': 0.3, 'translate_text': 0.5, 'analyze_sentiment': 0.6,
            'run_analysis': 0.8, 'generate_report': 0.9, 'optimize_model': 1.0
        }.get(service_type, 0.5)
        
        return np.array([service_complexity, urgency, budget/100.0, time_of_day])
    
    def predict_reward(self, agent: str, context: np.ndarray) -> float:
        """Predict reward for agent given context"""
        return np.dot(self.weights[agent], context)
    
    def select_agent(self, context: np.ndarray, exploration: float = 0.1) -> str:
        """Select agent based on predicted rewards"""
        if random.random() < exploration:
            return random.choice(self.agents)
        
        predictions = {agent: self.predict_reward(agent, context) 
                      for agent in self.agents}
        return max(predictions, key=predictions.get)
    
    def update_weights(self, agent: str, context: np.ndarray, reward: float, 
                      learning_rate: float = 0.01):
        """Update agent weights based on observed reward"""
        predicted = self.predict_reward(agent, context)
        error = reward - predicted
        self.weights[agent] += learning_rate * error * context
        
        self.history.append({
            'agent': agent,
            'context': context.tolist(),
            'reward': reward,
            'error': error,
            'timestamp': time.time()
        })

# Global bandit instances
service_bandit = MultiArmedBandit(['DataProcessor_A', 'Translator_B', 'Computer_C'])
contextual_bandit = ContextualBandit(['DataProcessor_A', 'Translator_B', 'Computer_C'])
```

### 16. ai/service_matcher.py
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple, Optional
import pandas as pd

class ServiceMatcher:
    def __init__(self):
        self.agent_features = {}  # Store agent performance features
        self.service_history = []
        self.models = {}  # ML models for different services
        self.is_trained = False
        
    def add_agent_features(self, agent_name: str, features: Dict):
        """Add/update agent features for matching"""
        self.agent_features[agent_name] = {
            'success_rate': features.get('success_rate', 0.9),
            'avg_response_time': features.get('response_time', 2.0),
            'reputation': features.get('reputation', 3.0),
            'price_competitiveness': features.get('price_competitiveness', 0.5),
            'specialization_score': features.get('specialization_score', 0.5),
            'availability': features.get('availability', 0.8)
        }
    
    def record_service_outcome(self, agent_name: str, service_type: str, 
                             client_satisfaction: float, completion_time: float,
                             price_paid: int, market_price: int):
        """Record service outcome for training"""
        if agent_name in self.agent_features:
            features = self.agent_features[agent_name].copy()
            features.update({
                'service_type': service_type,
                'price_ratio': price_paid / max(market_price, 1),
                'completion_time': completion_time,
                'satisfaction': client_satisfaction
            })
            self.service_history.append(features)
    
    def train_models(self):
        """Train ML models for service matching"""
        if len(self.service_history) < 10:  # Need minimum data
            return False
        
        df = pd.DataFrame(self.service_history)
        
        # Prepare features and targets
        feature_cols = ['success_rate', 'avg_response_time', 'reputation', 
                       'price_competitiveness', 'specialization_score', 
                       'availability', 'price_ratio']
        
        X = df[feature_cols].fillna(0)
        
        # Train different models for different criteria
        # 1. Overall satisfaction prediction
        y_satisfaction = (df['satisfaction'] > 0.7).astype(int)
        self.models['satisfaction'] = RandomForestClassifier(n_estimators=50, random_state=42)
        self.models['satisfaction'].fit(X, y_satisfaction)
        
        # 2. Fast completion prediction
        y_fast = (df['completion_time'] < df['completion_time'].median()).astype(int)
        self.models['speed'] = LogisticRegression(random_state=42)
        self.models['speed'].fit(X, y_fast)
        
        # 3. Value for money prediction
        df['value_score'] = df['satisfaction'] / (df['price_ratio'] + 0.1)
        y_value = (df['value_score'] > df['value_score'].median()).astype(int)
        self.models['value'] = RandomForestClassifier(n_estimators=30, random_state=42)
        self.models['value'].fit(X, y_value)
        
        self.is_trained = True
        return True
    
    def rank_agents(self, available_agents: List[str], service_type: str, 
                   criteria: str = 'satisfaction', market_price: int = 10) -> List[Tuple[str, float]]:
        """Rank agents based on criteria using ML predictions"""
        if not self.is_trained or criteria not in self.models:
            # Fallback to simple ranking
            return self._simple_ranking(available_agents, criteria)
        
        agent_scores = []
        for agent in available_agents:
            if agent in self.agent_features:
                features = self.agent_features[agent].copy()
                # Add contextual features
                features['price_ratio'] = features.get('base_price', market_price) / market_price
                
                feature_vector = np.array([[
                    features['success_rate'],
                    features['avg_response_time'],
                    features['reputation'],
                    features['price_competitiveness'],
                    features['specialization_score'],
                    features['availability'],
                    features['price_ratio']
                ]])
                
                # Get prediction probability
                score = self.models[criteria].predict_proba(feature_vector)[0][1]
                agent_scores.append((agent, score))
        
        return sorted(agent_scores, key=lambda x: x[1], reverse=True)
    
    def _simple_ranking(self, available_agents: List[str], criteria: str) -> List[Tuple[str, float]]:
        """Simple fallback ranking when ML models aren't available"""
        agent_scores = []
        for agent in available_agents:
            if agent in self.agent_features:
                features = self.agent_features[agent]
                if criteria == 'satisfaction':
                    score = (features['success_rate'] * 0.4 + 
                            features['reputation'] / 5.0 * 0.6)
                elif criteria == 'speed':
                    score = 1.0 / (features['avg_response_time'] + 0.1)
                elif criteria == 'value':
                    score = (features['success_rate'] * features['price_competitiveness'])
                else:
                    score = features['reputation'] / 5.0
                
                agent_scores.append((agent, score))
        
        return sorted(agent_scores, key=lambda x: x[1], reverse=True)
    
    def get_recommendation(self, service_type: str, criteria: str = 'satisfaction') -> Optional[str]:
        """Get single best agent recommendation"""
        available_agents = list(self.agent_features.keys())
        if not available_agents:
            return None
        
        rankings = self.rank_agents(available_agents, service_type, criteria)
        return rankings[0][0] if rankings else None
    
    def get_matching_stats(self):
        """Get service matching statistics"""
        return {
            'agents_tracked': len(self.agent_features),
            'service_records': len(self.service_history),
            'models_trained': len(self.models),
            'is_trained': self.is_trained
        }

# Global matcher instance
service_matcher = ServiceMatcher()
```

### 17. storage/__init__.py
```python
# Empty file to make storage a package
```

### 18. storage/storage_interface.py
```python
import json
import hashlib
from typing import Dict, Any, Optional
import time

class MockIPFS:
    def __init__(self):
        self.storage = {}
        self.metadata = {}
        
    def add(self, content: Any, content_type: str = "json") -> str:
        """Add content to storage and return CID"""
        if content_type == "json":
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        
        # Generate content ID (hash)
        cid = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        
        self.storage[cid] = content_str
        self.metadata[cid] = {
            'content_type': content_type,
            'size': len(content_str),
            'timestamp': time.time(),
            'access_count': 0
        }
        
        return cid
    
    def get(self, cid: str) -> Optional[Any]:
        """Retrieve content by CID"""
        if cid in self.storage:
            self.metadata[cid]['access_count'] += 1
            content_str = self.storage[cid]
            content_type = self.metadata[cid]['content_type']
            
            if content_type == "json":
                try:
                    return json.loads(content_str)
                except json.JSONDecodeError:
                    return content_str
            return content_str
        return None
    
    def pin(self, cid: str) -> bool:
        """Pin content (mark as important)"""
        if cid in self.metadata:
            self.metadata[cid]['pinned'] = True
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = sum(meta['size'] for meta in self.metadata.values())
        total_access = sum(meta['access_count'] for meta in self.metadata.values())
        
        return {
            'total_objects': len(self.storage),
            'total_size_bytes': total_size,
            'total_accesses': total_access,
            'pinned_objects': sum(1 for meta in self.metadata.values() 
                                if meta.get('pinned', False))
        }

# Global storage instance
mock_ipfs = MockIPFS()
```

### 19. blockchain/__init__.py
```python
# Empty file to make blockchain a package
```

### 20. blockchain/token_contract.sol
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract A2AToken {
    string public name = "A2A Marketplace Token";
    string public symbol = "A2A";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    // Service payment tracking
    mapping(bytes32 => ServicePayment) public servicePayments;
    
    struct ServicePayment {
        address client;
        address provider;
        uint256 amount;
        string serviceType;
        bool completed;
        uint256 timestamp;
    }
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event ServicePaid(bytes32 indexed paymentId, address indexed client, 
                     address indexed provider, uint256 amount, string serviceType);
    event ServiceCompleted(bytes32 indexed paymentId);
    
    constructor(uint256 _initialSupply) {
        totalSupply = _initialSupply * 10 ** decimals;
        balanceOf[msg.sender] = totalSupply;
    }
    
    function transfer(address _to, uint256 _value) public returns (bool) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        require(_to != address(0), "Invalid address");
        
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
    
    function approve(address _spender, uint256 _value) public returns (bool) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
    
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Insufficient allowance");
        require(_to != address(0), "Invalid address");
        
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        
        emit Transfer(_from, _to, _value);
        return true;
    }
    
    function payForService(address _provider, uint256 _amount, 
                          string memory _serviceType) public returns (bytes32) {
        require(balanceOf[msg.sender] >= _amount, "Insufficient balance");
        require(_provider != address(0), "Invalid provider address");
        
        bytes32 paymentId = keccak256(abi.encodePacked(
            msg.sender, _provider, _amount, _serviceType, block.timestamp
        ));
        
        // Hold funds in escrow
        balanceOf[msg.sender] -= _amount;
        
        servicePayments[paymentId] = ServicePayment({
            client: msg.sender,
            provider: _provider,
            amount: _amount,
            serviceType: _serviceType,
            completed: false,
            timestamp: block.timestamp
        });
        
        emit ServicePaid(paymentId, msg.sender, _provider, _amount, _serviceType);
        return paymentId;
    }
    
    function completeService(bytes32 _paymentId) public {
        ServicePayment storage payment = servicePayments[_paymentId];
        require(payment.provider == msg.sender, "Only provider can complete");
        require(!payment.completed, "Already completed");
        
        payment.completed = true;
        balanceOf[payment.provider] += payment.amount;
        
        emit ServiceCompleted(_paymentId);
        emit Transfer(address(this), payment.provider, payment.amount);
    }
    
    function refundService(bytes32 _paymentId) public {
        ServicePayment storage payment = servicePayments[_paymentId];
        require(payment.client == msg.sender, "Only client can request refund");
        require(!payment.completed, "Service already completed");
        require(block.timestamp > payment.timestamp + 1 hours, "Too early for refund");
        
        payment.completed = true; // Mark as resolved
        balanceOf[payment.client] += payment.amount;
        
        emit Transfer(address(this), payment.client, payment.amount);
    }
}
```

### 21. blockchain/voting_contract.sol
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AgentReputation {
    struct Agent {
        string name;
        uint256 totalRating;
        uint256 ratingCount;
        bool registered;
    }
    
    mapping(address => Agent) public agents;
    mapping(address => mapping(address => bool)) public hasRated;
    
    event AgentRegistered(address indexed agent, string name);
    event AgentRated(address indexed agent, address indexed rater, uint8 rating);
    
    function registerAgent(string memory _name) public {
        require(!agents[msg.sender].registered, "Agent already registered");
        
        agents[msg.sender] = Agent({
            name: _name,
            totalRating: 0,
            ratingCount: 0,
            registered: true
        });
        
        emit AgentRegistered(msg.sender, _name);
    }
    
    function rateAgent(address _agent, uint8 _rating) public {
        require(agents[_agent].registered, "Agent not registered");
        require(_rating >= 1 && _rating <= 5, "Rating must be 1-5");
        require(!hasRated[_agent][msg.sender], "Already rated this agent");
        require(_agent != msg.sender, "Cannot rate yourself");
        
        agents[_agent].totalRating += _rating;
        agents[_agent].ratingCount += 1;
        hasRated[_agent][msg.sender] = true;
        
        emit AgentRated(_agent, msg.sender, _rating);
    }
    
    function getAgentRating(address _agent) public view returns (uint256, uint256) {
        Agent memory agent = agents[_agent];
        if (agent.ratingCount == 0) {
            return (0, 0);
        }
        return (agent.totalRating, agent.ratingCount);
    }
    
    function getAverageRating(address _agent) public view returns (uint256) {
        Agent memory agent = agents[_agent];
        if (agent.ratingCount == 0) {
            return 0;
        }
        return (agent.totalRating * 100) / agent.ratingCount; // Multiply by 100 for precision
    }
}
```

### 22. ui/__init__.py
```python
# Empty file to make ui a package
```

### 23. ui/streamlit_ui.py
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agent_a import agent_a
from agents.agent_b import agent_b
from agents.agent_c import agent_c
from ledger.mock_ledger import ledger_instance
from ai.bandit_selection import service_bandit
from ai.rl_negotiation import global_negotiator
from communication.p2p_discovery import p2p_network

# Page configuration
st.set_page_config(
    page_title="A2A Economy Dashboard", 
    page_icon="🤖", 
    layout="wide"
)

# Initialize agents in P2P network
agents = {
    "DataProcessor_A": agent_a,
    "Translator_B": agent_b,
    "Computer_C": agent_c
}

for name, agent in agents.items():
    p2p_network.register(name, agent.list_services(), {
        'reputation': agent.reputation,
        'response_time': agent.response_time
    })

# Dashboard Header
st.title("🤖 A2A Economy Dashboard")
st.markdown("### Real-time Agent-to-Agent Marketplace Analytics")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
view_mode = st.sidebar.selectbox(
    "Select View", 
    ["Overview", "Agent Details", "Market Analytics", "ML Performance", "Network Stats"]
)

# Main content based on view mode
if view_mode == "Overview":
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Agents", 
            len(agents),
            delta=None
        )
    
    with col2:
        total_transactions = len(ledger_instance.get_transaction_history())
        st.metric(
            "Total Transactions", 
            total_transactions,
            delta="+2" if total_transactions > 0 else None
        )
    
    with col3:
        network_stats = p2p_network.get_network_stats()
        st.metric(
            "Services Available", 
            network_stats['unique_services'],
            delta=None
        )
    
    with col4:
        total_volume = sum(
            tx.amount for tx in ledger_instance.get_transaction_history()
        )
        st.metric(
            "Transaction Volume", 
            f"{total_volume} tokens",
            delta=f"+{total_volume//10}" if total_volume > 0 else None
        )
    
    # Agent Balances
    st.subheader("💰 Agent Balances")
    balance_data = []
    for agent_name in agents.keys():
        balance = ledger_instance.get_balance(agent_name)
        balance_data.append({"Agent": agent_name, "Balance": balance})
    
    df_balances = pd.DataFrame(balance_data)
    fig_balance = px.bar(
        df_balances, 
        x="Agent", 
        y="Balance", 
        title="Current Token Balances",
        color="Balance",
        color_continuous_scale="viridis"
    )
    st.plotly_chart(fig_balance, use_container_width=True)
    
    # Service Offerings
    st.subheader("🔧 Available Services")
    service_data = []
    for agent_name, agent in agents.items():
        for service, details in agent.list_services().items():
            service_data.append({
                "Agent": agent_name,
                "Service": service,
                "Price": details['price'],
                "Success Rate": details['success_rate'],
                "Reputation": details['reputation']
            })
    
    df_services = pd.DataFrame(service_data)
    st.dataframe(df_services, use_container_width=True)

elif view_mode == "Agent Details":
    st.subheader("🤖 Agent Performance Details")
    
    selected_agent = st.selectbox("Select Agent", list(agents.keys()))
    agent = agents[selected_agent]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Agent Information**")
        st.write(f"Name: {agent.name}")
        st.write(f"Balance: {ledger_instance.get_balance(agent.name)} tokens")
        st.write(f"Success Rate: {agent.success_rate:.2%}")
        st.write(f"Reputation: {agent.reputation:.1f}/5.0")
        st.write(f"Response Time: {agent.response_time:.1f}s")
        st.write(f"Current Load: {agent.load}")
    
    with col2:
        st.write("**Services Offered**")
        services_df = pd.DataFrame([
            {"Service": service, "Base Price": price}
            for service, price in agent.services.items()
        ])
        st.dataframe(services_df)
    
    # Transaction History
    st.write("**Transaction History**")
    agent_transactions = ledger_instance.get_transaction_history(agent.name)
    if agent_transactions:
        tx_data = []
        for tx in agent_transactions[-10:]:  # Last 10 transactions
            tx_data.append({
                "Transaction ID": tx.tx_id,
                "Type": "Received" if tx.receiver == agent.name else "Sent",
                "Amount": tx.amount,
                "Other Party": tx.sender if tx.receiver == agent.name else tx.receiver,
                "Service": tx.service or "N/A",
                "Timestamp": datetime.fromtimestamp(tx.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            })
        st.dataframe(pd.DataFrame(tx_data))
    else:
        st.info("No transaction history available")

elif view_mode == "Market Analytics":
    st.subheader("📈 Market Analytics")
    
    # Price Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Service Price Comparison**")
        all_services = set()
        for agent in agents.values():
            all_services.update(agent.services.keys())
        
        price_comparison = []
        for service in all_services:
            for agent_name, agent in agents.items():
                if service in agent.services:
                    price_comparison.append({
                        "Service": service,
                        "Agent": agent_name,
                        "Price": agent.services[service]
                    })
        
        if price_comparison:
            df_prices = pd.DataFrame(price_comparison)
            fig_prices = px.bar(
                df_prices, 
                x="Service", 
                y="Price", 
                color="Agent",
                title="Service Pricing by Agent",
                barmode="group"
            )
            st.plotly_chart(fig_prices, use_container_width=True)
    
    with col2:
        st.write("**Transaction Volume Over Time**")
        transactions = ledger_instance.get_transaction_history()
        if transactions:
            # Group transactions by hour
            tx_by_time = {}
            for tx in transactions:
                hour = datetime.fromtimestamp(tx.timestamp).strftime("%H:00")
                tx_by_time[hour] = tx_by_time.get(hour, 0) + tx.amount
            
            df_volume = pd.DataFrame([
                {"Time": time, "Volume": volume} 
                for time, volume in tx_by_time.items()
            ])
            
            fig_volume = px.line(
                df_volume, 
                x="Time", 
                y="Volume",
                title="Transaction Volume by Time"
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.info("No transaction data available")

elif view_mode == "ML Performance":
    st.subheader("🧠 Machine Learning Performance")
    
    # Bandit Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Multi-Armed Bandit Stats**")
        bandit_stats = service_bandit.get_stats()
        df_bandit = pd.DataFrame([
            {
                "Agent": agent,
                "Selections": stats["selections"],
                "Avg Reward": stats["avg_reward"],
                "Total Reward": stats["total_reward"]
            }
            for agent, stats in bandit_stats.items()
        ])
        st.dataframe(df_bandit)
        
        if not df_bandit.empty:
            fig_bandit = px.bar(
                df_bandit,
                x="Agent",
                y="Avg Reward",
                title="Average Reward by Agent"
            )
            st.plotly_chart(fig_bandit, use_container_width=True)
    
    with col2:
        st.write("**RL Negotiation Performance**")
        negotiation_stats = global_negotiator.get_strategy_stats()
        
        if "action_preferences" in negotiation_stats:
            df_actions = pd.DataFrame([
                {"Action": action, "Frequency": freq}
                for action, freq in negotiation_stats["action_preferences"].items()
            ])
            
            fig_actions = px.pie(
                df_actions,
                values="Frequency",
                names="Action",
                title="Negotiation Action Preferences"
            )
            st.plotly_chart(fig_actions, use_container_width=True)
        
        st.write(f"**States Learned:** {negotiation_stats.get('states_learned', 0)}")
        st.write(f"**Exploration Rate:** {negotiation_stats.get('exploration_rate', 0):.2%}")

elif view_mode == "Network Stats":
    st.subheader("🌐 Network Statistics")
    
    # Network Overview
    network_stats = p2p_network.get_network_stats()
    ledger_stats = ledger_instance.get_ledger_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**P2P Network**")
        st.write(f"Total Agents: {network_stats['total_agents']}")
        st.write(f"Online Agents: {network_stats['online_agents']}")
        st.write(f"Unique Services: {network_stats['unique_services']}")
    
    with col2:
        st.write("**Ledger Statistics**")
        st.write(f"Total Accounts: {ledger_stats['total_accounts']}")
        st.write(f"Total Supply: {ledger_stats['total_supply']} tokens")
        st.write(f"Total Transactions: {ledger_stats['total_transactions']}")
    
    with col3:
        st.write("**System Health**")
        st.write("🟢 All Systems Operational")
        st.write(f"📊 {len(agents)} Agents Active")
        st.write(f"💰 {ledger_stats['locked_funds']} Escrow Transactions")
    
    # Service Distribution
    st.write("**Service Distribution**")
    service_count = {}
    for agent in agents.values():
        for service in agent.services:
            service_count[service] = service_count.get(service, 0) + 1
    
    df_service_dist = pd.DataFrame([
        {"Service": service, "Providers": count}
        for service, count in service_count.items()
    ])
    
    fig_service_dist = px.bar(
        df_service_dist,
        x="Service",
        y="Providers",
        title="Number of Providers per Service"
    )
    st.plotly_chart(fig_service_dist, use_container_width=True)

# Real-time updates
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Footer
st.markdown("---")
st.markdown("🤖 A2A Marketplace Simulator - Built with Streamlit")
```

### 24. ui/interaction_ui.py
```python
import streamlit as st
import time
import random
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agent_a import agent_a
from agents.agent_b import agent_b
from agents.agent_c import agent_c
from ai.rl_negotiation import global_negotiator
from ai.bandit_selection import service_bandit, contextual_bandit
from communication.message_schema import ServiceRequest
from communication.p2p_discovery import p2p_network
from ledger.mock_ledger import ledger_instance
import numpy as np

# Page configuration
st.set_page_config(
    page_title="A2A Interaction Simulator", 
    page_icon="🔄", 
    layout="centered"
)

# Initialize agents
agents = {
    "DataProcessor_A": agent_a,
    "Translator_B": agent_b,
    "Computer_C": agent_c
}

# Register agents in P2P network
for name, agent in agents.items():
    p2p_network.register(name, agent.list_services())

st.title("🔄 A2A Economy Interaction Simulator")
st.markdown("### Simulate autonomous agent interactions with AI-powered negotiation")

# Simulation Mode Selection
st.sidebar.header("🎮 Simulation Controls")
simulation_mode = st.sidebar.selectbox(
    "Simulation Mode",
    ["Manual Request", "Auto Simulation", "Batch Processing"]
)

if simulation_mode == "Manual Request":
    st.subheader("🎯 Manual Service Request")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sender = st.selectbox("🤖 Requester Agent", options=list(agents.keys()))
        
    with col2:
        # Get available receivers (excluding sender)
        available_receivers = [name for name in agents.keys() if name != sender]
        receiver = st.selectbox("🎯 Provider Agent", options=available_receivers)
    
    # Get services from selected receiver
    if receiver:
        receiver_agent = agents[receiver]
        available_services = list(receiver_agent.services.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            service_type = st.selectbox("🔧 Service Type", options=available_services)
            
        with col2:
            base_price = receiver_agent.services.get(service_type, 10)
            offered_price = st.number_input(
                "💰 Offered Price", 
                min_value=1, 
                max_value=base_price * 2,
                value=base_price,
                step=1
            )
    
    # Advanced Options
    with st.expander("⚙️ Advanced Options"):
        use_negotiation = st.checkbox("🤝 Enable AI Negotiation", value=True)
        use_bandit = st.checkbox("🎰 Use Bandit Selection", value=True)
        urgency = st.slider("⏰ Urgency Level", 0.0, 1.0, 0.5, 0.1)
    
    # Submit Request
    if st.button("📤 Submit Request", type="primary"):
        with st.spinner("🔄 Processing request..."):
            # Create service request
            request = ServiceRequest(
                sender=sender,
                receiver=receiver,
                service_type=service_type,
                offered_price=offered_price
            )
            
            st.markdown("## 📋 Processing Steps")
            
            # Step 1: Service Discovery
            st.write("**1. 🔍 Service Discovery**")
            providers = p2p_network.discover(service_type)
            st.success(f"Found {len(providers)} providers for '{service_type}'")
            
            # Step 2: Agent Selection (if bandit enabled)
            if use_bandit and len(providers) > 1:
                st.write("**2. 🎰 Bandit Agent Selection**")
                
                # Use contextual bandit
                context = contextual_bandit.get_context(
                    service_type, urgency, offered_price, 
                    time.time() % 24 / 24  # time of day
                )
                
                selected_agent = contextual_bandit.select_agent(context)
                st.info(f"Bandit selected: {selected_agent}")
                
                if selected_agent != receiver:
                    st.warning(f"Redirecting from {receiver} to {selected_agent}")
                    receiver = selected_agent
                    receiver_agent = agents[receiver]
                    request.receiver = receiver
            
            # Step 3: Negotiation Phase
            if use_negotiation:
                st.write("**3. 🤝 AI Negotiation Phase**")
                
                market_price = receiver_agent.services[service_type]
                agent_reputation = receiver_agent.reputation
                
                negotiation_result, final_price = global_negotiator.negotiate(
                    service_type, offered_price, market_price, agent_reputation
                )
                
                st.write(f"Negotiation outcome: **{negotiation