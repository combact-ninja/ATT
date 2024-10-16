import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Define CNN model
class DiseaseCNN(nn.Module):
    def __init__(self):
        super(DiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: diseased or healthy

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Define a custom dataset class for the disease detection task
class DiseaseDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=(1, 32, 32)):
        self.num_samples = num_samples
        self.img_size = img_size

        # Random image data (representing medical images)
        self.states = torch.randn(num_samples, *img_size)

        # Random next states (for simplicity, same as current states)
        self.next_states = self.states.clone()

        # Random actions (0 for healthy, 1 for diseased)
        self.actions = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        state = self.states[idx]
        next_state = self.next_states[idx]
        action = self.actions[idx]
        return state, next_state, action


# TD learning update function with reward calculation
def td_update(model, optimizer, state, action, next_state, gamma=0.99):
    model.train()

    # Forward pass for current state
    pred = model(state)

    # Get predicted action (class with highest score)
    predicted_action = torch.argmax(pred, dim=1)

    # Calculate reward (1 for correct, -1 for incorrect)
    reward = (predicted_action == action).float() * 2 - 1

    # Get predicted value for the chosen action
    predicted_value = pred.gather(1, action.unsqueeze(1)).squeeze()

    # Forward pass for next state
    next_pred = model(next_state).detach()

    # Calculate target using TD formula
    target = reward + gamma * torch.max(next_pred, dim=1)[0]

    # TD error
    td_error = target - predicted_value

    # Loss function (MSE of TD error)
    loss = td_error.pow(2).mean()

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Modified training loop with dynamic reward calculation
def train_td_cnn(model, dataloader, num_episodes=10, gamma=0.99):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for episode in range(num_episodes):
        total_loss = 0
        for batch in dataloader:
            state, next_state, action = batch
            loss = td_update(model, optimizer, state, action, next_state, gamma)
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        print(f'Episode {episode + 1}/{num_episodes}, Avg Loss: {avg_loss:.4f}')


# Initialize the model
model = DiseaseCNN()

# Create the DataLoader
dataset = DiseaseDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the CNN using TD learning
train_td_cnn(model, dataloader, num_episodes=10, gamma=0.99)
