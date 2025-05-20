import streamlit as st
import requests
import tempfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
import time
import traceback  # Add this import to fix traceback error
import subprocess
import platform
import socket

# Define room colors for frontend display
room_colors = {
    "Living Room": "#8DD3C7",  # Teal
    "Bedroom": "#FFFFB3",      # Light yellow
    "Bathroom": "#BEBADA",     # Lavender
    "Kitchen": "#FB8072",      # Salmon pink
    "Room": "#80B1D3"          # Light blue
}

# ✅ Final backend URL
BACKEND_URL = "https://track-3d-4.onrender.com"

# Function to check if backend is running and reachable
def check_backend_connection(url=BACKEND_URL, retry_count=2, timeout=3):
    """Check if backend server is running and reachable"""
    for attempt in range(retry_count):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True, "Connected"
            else:
                return False, f"Backend returned status code {response.status_code}"
        except requests.exceptions.ConnectionError:
            if attempt < retry_count - 1:
                time.sleep(1)
                continue
            return False, "Connection refused"
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                continue
            return False, "Connection timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"
    return False, "Could not connect after retries"

# Function to display project information in the sidebar
def show_sidebar_info():
    """Display project information in the sidebar"""
    st.sidebar.markdown("### 🏡 About This Project")
    
    st.sidebar.markdown("""
    **House Construction Cost Predictor** is an AI-powered tool that helps you:
    
    - ✅ **Detect walls and rooms** from floorplans  
    - ✅ **Identify room types** automatically  
    - ✅ **Generate 3D models** of your floorplans  
    - ✅ **Estimate construction costs** based on area and features  
    
    Perfect for architects, builders, and homeowners planning construction projects.
    """)
    
    st.sidebar.markdown("---")
    
    connected, message = check_backend_connection()
    if connected:
        st.sidebar.success("✅ Backend Connected")
    else:
        st.sidebar.error(f"❌ Backend Unreachable: {message}")

# Page config and title
st.set_page_config(page_title="🏡 House Construction Cost Predictor & 3D Viewer")
st.title("🏡 House Construction Cost Predictor & 3D Model Viewer")

# Sidebar content
show_sidebar_info()

# File uploader
st.markdown("""
### Upload your floorplan image
- Click the **large + button** or **drag and drop** your file here.
- Supported formats: PNG, JPG, JPEG
""")

uploaded_file = st.file_uploader(
    "Upload Floorplan Image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="fileUploader",
    label_visibility="visible"
)

# Message below uploader
if not uploaded_file:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; margin-bottom: 30px;">
            <p style="color: #757575; font-size: 0.9em; font-style: italic;">
                ✨ Turning floorplans into 3D dreams with AI ❤️
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("1️⃣ Detect Walls and Rooms"):
            st.write("---")
            with st.spinner("Detecting walls and rooms..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_image_path = tmp_file.name

                    # Skip backend check on deployed version
                    connected, status_msg = check_backend_connection(
                        retry_count=3, timeout=5, url=BACKEND_URL
                    )
                    if not connected:
                        st.error(f"Could not connect to backend: {status_msg}")
                        
                        # Only offer restart if running locally
                        if platform.system() != "Linux":
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if st.button("🔄 Start Backend"):
                                    success, msg = try_start_backend_server()
                                    if success:
                                        st.info(f"{msg}. Please wait a moment and try again.")
                                        time.sleep(3)
                                        st.experimental_rerun()
                                    else:
                                        st.error(msg)
                            with col2:
                                st.code("cd c:\\Users\\USER\\Desktop\\vit_project\\model_folder && python app.py")
                        
                        st.stop()

                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            with open(temp_image_path, "rb") as f:
                                files = {"file": (os.path.basename(temp_image_path), f, "image/png")}
                                detection_response = requests.post(
                                    f"{BACKEND_URL}/detect-walls-rooms", 
                                    files=files, 
                                    timeout=120
                                )
                            break
                        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                            if attempt < max_retries - 1:
                                st.warning("Connection issue detected. Retrying...")
                                time.sleep(2)
                                continue
                            else:
                                raise

                    if detection_response.status_code == 200:
                        result = detection_response.json()
                        st.markdown("""
                        <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:15px;">
                            <h2 style="color:#1E88E5;margin-bottom:0;text-align:center;">🔍 Wall & Room Detection Results</h2>
                        </div>
                        """, unsafe_allow_html=True)

                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("🧱 Walls Detected", result['wall_count'])
                        with col_stat2:
                            st.metric("🏠 Rooms Detected", result['room_count'])

                        vis_url = f"{BACKEND_URL}{result['visualization']}"

                        try:
                            st.image(vis_url, caption="Wall and Room Detection Results", use_container_width=True, width=800)
                        except Exception as img_error:
                            st.warning(f"Could not load visualization image. Error: {img_error}")
                            try:
                                with st.spinner("Downloading visualization image..."):
                                    vis_response = requests.get(vis_url, timeout=30)
                                    if vis_response.status_code == 200:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_tmp:
                                            img_tmp.write(vis_response.content)
                                            img_path = img_tmp.name
                                        st.image(img_path, caption="Wall and Room Detection Results", use_container_width=True)
                                        try:
                                            os.unlink(img_path)
                                        except:
                                            pass
                                    else:
                                        st.error(f"Failed to download visualization: HTTP {vis_response.status_code}")
                            except Exception as local_err:
                                st.error(f"Could not display visualization: {local_err}")

                        st.session_state.wall_count = result['wall_count']
                        st.session_state.room_count = result['room_count']

                        # The rest of your room-type breakdown, tabs, and feature display remain unchanged

                        st.success("✅ Wall and room detection complete! You can now proceed to generate the 3D model.")

                        if st.button("↩️ Return to Selection"):
                            st.experimental_rerun()

                    else:
                        try:
                            error_data = detection_response.json()
                            error_message = error_data.get('error', 'Unknown error')
                            st.error(f"Detection failed: {error_message}")
                            if 'traceback' in error_data:
                                with st.expander("Error Details", expanded=False):
                                    st.code(error_data['traceback'], language='python')
                        except:
                            st.error(f"Detection failed with status code {detection_response.status_code}: {detection_response.text}")

                    try:
                        os.remove(temp_image_path)
                    except Exception:
                        pass

                except requests.exceptions.Timeout:
                    st.error("⏱️ Wall detection request timed out. Try again later.")

                except requests.exceptions.ConnectionError:
                    st.error("🔌 Connection to backend was lost.")

                except Exception as e:
                    import traceback
                    st.error(f"❌ Wall detection error: {e}")
                    with st.expander("Error Details", expanded=False):
                        st.code(traceback.format_exc(), language='python')
    with col2:
     if st.button("2️⃣ Generate 3D Model"):
        st.write("---")
        
        with st.spinner("Generating 3D model, please wait..."):
            try:
                connected, status_msg = check_backend_connection(retry_count=3, timeout=5)
                if not connected:
                    st.error(f"Could not connect to backend: {status_msg}")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if st.button("🔄 Start Backend"):
                            success, msg = try_start_backend_server()
                            if success:
                                st.info(f"{msg}. Please wait a moment and try again.")
                                time.sleep(3)
                                st.experimental_rerun()
                            else:
                                st.error(msg)
                    with col2:
                        st.code("cd c:\\Users\\USER\\Desktop\\vit_project\\model_folder && python app.py")
                    st.stop()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_image_path = tmp_file.name

                with open(temp_image_path, "rb") as f:
                    files = {"file": (os.path.basename(temp_image_path), f, "image/png")}
                    response = requests.post("http://127.0.0.1:5000/generate-3d", files=files, timeout=180)

                if response.status_code == 200:
                    data = response.json()
                    glb_path = data.get("3d_model")  # Changed from ply_path
                    floorplan_data = data.get("floorplan_data", {})

                    if glb_path:
                        st.markdown("""
                        <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:15px;">
                            <h2 style="color:#1E88E5;margin-bottom:0;text-align:center;">🏗️ 3D Model & Construction Cost Results</h2>
                        </div>
                        """, unsafe_allow_html=True)

                        st.subheader("Detected Floorplan Features")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Total Rooms", floorplan_data.get("rooms", "N/A"))
                        with c2:
                            st.metric("Estimated Area (m²)", floorplan_data.get("estimated_area", "N/A"))
                        with c3:
                            room_counts = floorplan_data.get("room_counts", {})
                            st.metric("Bedrooms / Bathrooms", f"{room_counts.get('bedroom', 0)} / {room_counts.get('bathroom', 0)}")

                        area = floorplan_data.get("estimated_area", 0)
                        bedrooms = room_counts.get("bedroom", 0)
                        bathrooms = room_counts.get("bathroom", 0)
                        kitchens = room_counts.get("kitchen", 0)
                        living = room_counts.get("living", 0)

                        if area > 0:
                            st.subheader("Estimated Construction Cost")
                            location_factor = {
                                "Urban (High Cost)": 1.3,
                                "Suburban": 1.0,
                                "Rural (Low Cost)": 0.8
                            }[st.selectbox("Location Type", list(location_factor.keys()))]

                            try:
                                cost_response = requests.post(
                                    "http://127.0.0.1:5000/estimate-construction-cost",
                                    json={
                                        "area": area,
                                        "bedrooms": bedrooms,
                                        "bathrooms": bathrooms,
                                        "kitchen": kitchens,
                                        "living": living,
                                        "location_factor": location_factor
                                    },
                                    timeout=10
                                )
                                if cost_response.status_code == 200:
                                    cost_data = cost_response.json()
                                    estimated_cost = cost_data.get("estimated_cost", 0)
                                    breakdown = cost_data.get("breakdown", {})
                                    st.success(f"🏗️ Total Estimated Construction Cost: ${estimated_cost:,.2f}")

                                    with st.expander("Cost Breakdown"):
                                        st.write(f"Base Area Cost: ${breakdown.get('base_area_cost', 0):,.2f}")
                                        st.write(f"Bedroom Cost: ${breakdown.get('bedroom_cost', 0):,.2f}")
                                        st.write(f"Bathroom Cost: ${breakdown.get('bathroom_cost', 0):,.2f}")
                                        st.write(f"Kitchen Cost: ${breakdown.get('kitchen_cost', 0):,.2f}")
                                        st.write(f"Living Room Cost: ${breakdown.get('living_cost', 0):,.2f}")
                                else:
                                    st.error(f"Failed to estimate cost: {cost_response.text}")
                            except Exception as e:
                                st.error(f"Error estimating construction cost: {e}")

                        # GLB file viewer and download
                        if glb_path.startswith("/static/"):
                            glb_url = f"http://127.0.0.1:5000{glb_path}"
                        else:
                            glb_url = f"http://127.0.0.1:5000/static/{os.path.basename(glb_path)}"

                        glb_content = requests.get(glb_url).content
                        try:
                            st.download_button("Download 3D Model (.glb)", data=glb_content, file_name="model.glb", mime="model/gltf-binary")
                        except Exception as e:
                            st.error(f"Download error: {e}")

                        st.markdown("### 3D Model Viewer")

                        glb_base64 = base64.b64encode(glb_content).decode('utf-8')

                        viewer_html = f"""
                        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
                        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

                        <div id="glb-viewer" style="width:100%;height:500px;background:#111;"></div>
                        <script>
                            const container = document.getElementById("glb-viewer");
                            const scene = new THREE.Scene();
                            const camera = new THREE.PerspectiveCamera(75, container.clientWidth/container.clientHeight, 0.1, 1000);
                            camera.position.set(2, 2, 5);

                            const renderer = new THREE.WebGLRenderer({antialias:true});
                            renderer.setSize(container.clientWidth, container.clientHeight);
                            container.appendChild(renderer.domElement);

                            const light = new THREE.HemisphereLight(0xffffff, 0x444444);
                            light.position.set(0, 20, 0);
                            scene.add(light);

                            const controls = new THREE.OrbitControls(camera, renderer.domElement);

                            const glbData = Uint8Array.from(atob("{glb_base64}"), c => c.charCodeAt(0)).buffer;
                            const loader = new THREE.GLTFLoader();
                            loader.parse(glbData, '', gltf => {{
                                scene.add(gltf.scene);
                            }}, err => console.error(err));

                            function animate() {{
                                requestAnimationFrame(animate);
                                controls.update();
                                renderer.render(scene, camera);
                            }}
                            animate();
                        </script>
                        """
                        try:
                            st.components.v1.html(viewer_html, height=520)
                            st.info("You can also open the .glb file in software like Blender or Windows 3D Viewer.")
                        except Exception as e:
                            st.error(f"Viewer error: {e}")

                        if st.button("↩️ Return to Selection"):
                            st.experimental_rerun()
                    else:
                        st.error("3D model not found.")
                else:
                    try:
                        err = response.json()
                        st.error(f"3D model generation failed: {err.get('error', response.text)}")
                    except:
                        st.error(f"3D model generation failed: {response.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out.")
            except requests.exceptions.ConnectionError:
                st.error("Connection lost. Ensure Flask backend is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        try:
            os.remove(temp_image_path)
        except Exception:
            pass

def display_footer():
    st.markdown("---")
    footer_cols = st.columns([1, 3, 1])
    with footer_cols[1]:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <p style="color: #5A5A5A; font-size: 0.9em; margin-bottom: 5px;">
                Made with ❤️ for architects and home builders
            </p>
            <p style="color: #757575; font-size: 0.8em; font-style: italic;">
                Turn your floorplans into intelligent 3D models with AI
            </p>
            <p style="color: #9E9E9E; font-size: 0.7em; margin-top: 15px;">
                © 2025 House Construction Cost Predictor | All Rights Reserved
            </p>
        </div>
        """, unsafe_allow_html=True)

display_footer()
