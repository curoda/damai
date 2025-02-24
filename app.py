import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

# Configure the Streamlit page
st.set_page_config(page_title="24x24 g Matrix Calculator", layout="wide")

st.title("24x24 g Matrix Calculator (New Instructions)")

st.markdown(r"""
**New Formula**:
\[
g(l,m,n,p,q,r) \;=\; 
\cos\!\Bigl(\frac{2\pi}{6800}\,(p\,l + q\,m + r\,n)\Bigr)
\;-\;
i\,\sin\!\Bigl(\frac{2\pi}{6800}\,(p\,l + q\,m + r\,n)\Bigr).
\]

We have a single table with 24 rows, each row containing \((l,m,n,p,q,r)\).  

- The **column** index \(j\) of the 24×24 matrix corresponds to \((l_j,m_j,n_j)\).  
- The **row** index \(i\) corresponds to \((p_i,q_i,r_i)\).  

Thus, for \((i,j)\) we do:
\[
g_{i,j} \;=\;
\cos\!\Bigl(\tfrac{2\pi}{6800}\,\bigl[p_i\,l_j + q_i\,m_j + r_i\,n_j\bigr]\Bigr)
\;-\;
i\,\sin\!\Bigl(\tfrac{2\pi}{6800}\,\bigl[p_i\,l_j + q_i\,m_j + r_i\,n_j\bigr]\Bigr).
\]

After building the matrix \(g\), we compute:
1. **gcc** = the complex conjugate of \(g\).
2. **gccᵀ** = the transpose of gcc.
3. The product \(g \times gcc^\mathsf{T}\) (standard matrix multiplication).
""")

st.markdown("### 1. Upload Your Spreadsheet")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

@st.cache_data
def load_data(file: BytesIO, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
    st.write("File Details:", file_details)
    
    # Read the file based on its extension
    if uploaded_file.name.endswith('.csv'):
        df = load_data(uploaded_file, "csv")
    else:
        df = load_data(uploaded_file, "xlsx")
    
    st.markdown("### 2. Input Data Preview")
    st.dataframe(df)
    
    # Check that the required columns exist.
    required_cols = ["index", "l", "m", "n", "p", "q", "r"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Uploaded file must contain the following columns: {required_cols}")
    else:
        # Ensure exactly 24 rows are present.
        if df.shape[0] != 24:
            st.warning("Warning: The data does not contain exactly 24 rows. The calculations assume 24 rows (indexed 1 to 24).")
        
        # Extract column values
        l_arr = df["l"].to_numpy()
        m_arr = df["m"].to_numpy()
        n_arr = df["n"].to_numpy()
        p_arr = df["p"].to_numpy()
        q_arr = df["q"].to_numpy()
        r_arr = df["r"].to_numpy()
        
        # Define function to compute each matrix element
        def compute_entry(p_val, q_val, r_val, l_val, m_val, n_val):
            product = p_val * l_val + q_val * m_val + r_val * n_val
            angle = (2 * np.pi / 6800.0) * product
            return np.cos(angle) - 1j * np.sin(angle)
        
        # Build the 24x24 g matrix
        # Row i uses (p_i, q_i, r_i)
        # Column j uses (l_j, m_j, n_j)
        G = np.empty((24, 24), dtype=complex)
        for i in range(24):
            for j in range(24):
                G[i, j] = compute_entry(
                    p_arr[i], q_arr[i], r_arr[i], 
                    l_arr[j], m_arr[j], n_arr[j]
                )
        
        # Compute the complex conjugate matrix gcc
        G_cc = np.conjugate(G)
        
        # Compute the transpose of gcc
        G_cc_T = G_cc.T
        
        # Standard matrix multiplication: g x gccᵀ
        G_product = G @ G_cc_T
        
        st.markdown("### 3. Results")
        
        with st.expander("Show g Matrix (24×24)"):
            st.dataframe(pd.DataFrame(G.astype(str)))
        
        with st.expander("Show Complex Conjugate Matrix (gcc) (24×24)"):
            st.dataframe(pd.DataFrame(G_cc.astype(str)))
        
        with st.expander("Show Transpose of gcc (gccᵀ) (24×24)"):
            st.dataframe(pd.DataFrame(G_cc_T.astype(str)))
        
        with st.expander("Show Product (g × gccᵀ) (24×24)"):
            st.dataframe(pd.DataFrame(G_product.astype(str)))
            
        st.success("Calculations complete!")
