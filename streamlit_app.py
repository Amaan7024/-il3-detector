import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ---------------- RamanNet Model ---------------- #
class RamanNet(nn.Module):
    def __init__(self, input_length=1000, window_size=50, step=25, n1=32, n2=256, embedding_dim=128, num_classes=2):
        super(RamanNet, self).__init__()
        self.window_size = window_size
        self.step = step
        self.num_windows = (input_length - window_size) // step + 1

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, n1),
                nn.BatchNorm1d(n1),
                nn.LeakyReLU()
            ) for _ in range(self.num_windows)
        ])

        self.dropout1 = nn.Dropout(0.4)
        self.summary_dense = nn.Sequential(
            nn.Linear(n1 * self.num_windows, n2),
            nn.BatchNorm1d(n2),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        self.embedding = nn.Sequential(
            nn.Linear(n2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        windows = []
        for i in range(self.num_windows):
            start = i * self.step
            end = start + self.window_size
            window = x[:, start:end]
            windows.append(self.blocks[i](window))
        x = torch.cat(windows, dim=1)
        x = self.dropout1(x)
        x = self.summary_dense(x)
        emb = nn.functional.normalize(self.embedding(x), p=2, dim=1)
        out = self.classifier(emb)
        return out, emb

# ---------------- Helper Functions ---------------- #
def preprocess_signal(df, input_len=1000, window_size=50, step=25):
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    x_uniform = np.linspace(x.min(), x.max(), input_len)
    y_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")(x_uniform)
    y_norm = StandardScaler().fit_transform(y_interp.reshape(-1, 1)).flatten()

    segments = []
    for i in range(0, len(y_norm) - window_size + 1, step):
        segments.append(y_norm[i:i + window_size])
    input_tensor = torch.tensor(np.concatenate(segments)).float().unsqueeze(0)
    return input_tensor

@st.cache_resource
def load_model(path="ramannet_model.pt"):
    model = RamanNet(input_length=1000)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="IL-3 Detection", layout="centered")
st.title("🧪 IL-3 Detection from Raman Spectrum")
st.markdown("Upload your Raman spectrum (.csv or .txt) to check if IL-3 is present.")

uploaded_file = st.file_uploader("📂 Upload Raman Spectrum File", type=["csv", "txt"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        with st.expander("📈 View Uploaded Spectrum"):
            fig, ax = plt.subplots()
            ax.plot(df.iloc[:, 0], df.iloc[:, 1], color='purple')
            ax.set_title("Raman Spectrum")
            ax.set_xlabel("Wavenumber (cm⁻\u00b9)")
            ax.set_ylabel("Intensity")
            st.pyplot(fig)

        # Preprocess and Predict
        input_tensor = preprocess_signal(df)
        model = load_model()
        with torch.no_grad():
            output, _ = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            prob = torch.softmax(output, dim=1)[0][pred].item()

        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        if prob < 0.65:
            st.warning(f"⚠️ Uncertain – Retest recommended")
            st.progress(int(prob * 100))
        elif pred == 1:
            st.success(f"✅ IL-3 Present")
            st.progress(int(prob * 100))
        else:
            st.error(f"❌ IL-3 Absent")
            st.progress(int(prob * 100))

        st.caption(f"Confidence: {prob:.2%}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
