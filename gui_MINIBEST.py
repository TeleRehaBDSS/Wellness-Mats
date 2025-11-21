import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from main import (
    load_basic_signals,
    process_compensatory_stepping,
    process_rise_to_toes,
    process_sit_to_stand,
    process_stand_on_one_leg,
    process_stance_eyes_open,
    process_stance_eyes_closed,
    _infer_participant_and_exercise,
    BasicMatSignals,
    _get_blobs
)

class ModernMiniBESTGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MiniBEST Mat Analysis Dashboard")
        self.geometry("1400x950")
        
        # Theme configuration
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('clam')
        except:
            pass
            
        self._configure_styles()

        # Application State
        self.current_csv_path = None
        self.current_signals = None
        self.current_result = None

        # Replay state (for stance exercises and others that use frame-by-frame video)
        self.replay_frames = None
        self.replay_ax = None
        self.replay_im = None
        self.replay_running = False
        self.replay_after_id = None
        self.replay_frame_index = 0
        self.replay_start_index = 0
        self.replay_skip = 1  # how many frames to advance per step (set for stance)
        self.replay_indices = None  # optional list of frame indices to use (e.g., only active frames)
        self.replay_time = None     # corresponding time stamps for frames
        self.replay_start_time = 0.0
        self.animation = None
        
        # Main Layout
        self._create_layout()
        
    def _configure_styles(self):
        # Colors
        self.bg_color = "#f5f6f7"
        self.panel_bg = "#ffffff"
        self.accent_color = "#0078d4" # Modern blue
        self.text_primary = "#201f1e"
        self.text_secondary = "#605e5c"
        
        self.configure(bg=self.bg_color)
        
        # General Styles
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("Card.TFrame", background=self.panel_bg, relief="flat")
        
        # Labels
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_primary, font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background=self.panel_bg, foreground=self.text_primary, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background=self.bg_color, foreground=self.accent_color, font=("Segoe UI", 18, "bold"))
        self.style.configure("SubHeader.TLabel", background=self.panel_bg, foreground=self.text_secondary, font=("Segoe UI", 12, "bold"))
        
        # Score Display
        self.style.configure("ScoreTitle.TLabel", background=self.panel_bg, foreground=self.text_secondary, font=("Segoe UI", 14))
        self.style.configure("ScoreValue.TLabel", background=self.panel_bg, foreground=self.accent_color, font=("Segoe UI", 64, "bold"))
        
        # Buttons
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        self.style.map("Accent.TButton", background=[('active', "#006cc1"), ('!disabled', self.accent_color)], foreground=[('!disabled', 'white')])

        # Treeview
        self.style.configure("Treeview", rowheight=30, font=("Segoe UI", 10), background="white", fieldbackground="white")
        self.style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _create_layout(self):
        # Use a PanedWindow for resizable layout
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- LEFT SIDEBAR (Controls & Score) ---
        self.sidebar = ttk.Frame(self.paned, width=350)
        self.paned.add(self.sidebar, weight=1)
        
        # Input Card
        self.input_card = ttk.Frame(self.sidebar, style="Card.TFrame", padding=20)
        self.input_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.input_card, text="Analysis Setup", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 15))
        
        # File Selection
        ttk.Label(self.input_card, text="Data File (CSV):", style="Card.TLabel").pack(anchor="w")
        self.file_frame = ttk.Frame(self.input_card, style="Card.TFrame")
        self.file_frame.pack(fill=tk.X, pady=(5, 15))
        
        self.lbl_filename = ttk.Label(self.file_frame, text="No file selected", style="Card.TLabel", wraplength=280, foreground="gray")
        self.lbl_filename.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self.file_frame, text="Browse...", command=self.browse_file).pack(side=tk.RIGHT)
        
        # Exercise Selection
        ttk.Label(self.input_card, text="Exercise Type:", style="Card.TLabel").pack(anchor="w")
        self.exercise_var = tk.StringVar()
        self.combo_exercise = ttk.Combobox(self.input_card, textvariable=self.exercise_var, state="readonly", height=15)
        self.combo_exercise['values'] = [
            "Compensatory Stepping (FORWARD)",
            "Compensatory Stepping (BACKWARD)",
            "Compensatory Stepping (LATERAL-LEFT)",
            "Compensatory Stepping (LATERAL-RIGHT)",
            "Rise to Toes",
            "Sit to Stand",
            "Stand on One Leg (Left)",
            "Stand on One Leg (Right)",
            "Stance (Eyes Open)",
            "Stance (Eyes Closed)",
        ]
        self.combo_exercise.pack(fill=tk.X, pady=(5, 15))
        self.combo_exercise.bind("<<ComboboxSelected>>", self._on_exercise_change)
        
        # Participant
        ttk.Label(self.input_card, text="Participant ID:", style="Card.TLabel").pack(anchor="w")
        self.entry_participant = ttk.Entry(self.input_card)
        self.entry_participant.pack(fill=tk.X, pady=(5, 20))
        
        # Analyze Button
        self.btn_analyze = ttk.Button(self.input_card, text="ANALYZE EXERCISE", style="Accent.TButton", command=self.on_analyze, cursor="hand2")
        self.btn_analyze.pack(fill=tk.X, ipady=10)
        
        # Score Card
        self.score_card = ttk.Frame(self.sidebar, style="Card.TFrame", padding=20)
        self.score_card.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.score_card, text="MiniBEST Score", style="ScoreTitle.TLabel").pack(anchor="center")
        self.lbl_score = ttk.Label(self.score_card, text="-", style="ScoreValue.TLabel")
        self.lbl_score.pack(anchor="center", pady=10)
        self.lbl_score_desc = ttk.Label(self.score_card, text="", style="Card.TLabel", foreground="gray")
        self.lbl_score_desc.pack(anchor="center")

        # --- RIGHT MAIN AREA (Metrics & Plots) ---
        self.main_area = ttk.Frame(self.paned)
        self.paned.add(self.main_area, weight=4)
        
        # Notebook for Tabs
        self.notebook = ttk.Notebook(self.main_area)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Dashboard (Plots + Key Metrics)
        self.tab_dashboard = ttk.Frame(self.notebook, style="Card.TFrame", padding=10)
        self.notebook.add(self.tab_dashboard, text="  Visual Analysis  ")
        
        # Split Dashboard: Top for Metrics Table, Bottom for Plots
        self.dash_paned = ttk.PanedWindow(self.tab_dashboard, orient=tk.VERTICAL)
        self.dash_paned.pack(fill=tk.BOTH, expand=True)
        
        # Metrics Table Area
        self.metrics_frame = ttk.Frame(self.dash_paned, style="Card.TFrame")
        self.dash_paned.add(self.metrics_frame, weight=1)
        
        self.tree_metrics = ttk.Treeview(self.metrics_frame, columns=("Value",), show="tree headings")
        self.tree_metrics.heading("#0", text="Feature", anchor="w")
        self.tree_metrics.heading("Value", text="Value", anchor="w")
        self.tree_metrics.column("#0", width=300)
        self.tree_metrics.column("Value", width=150)
        self.tree_metrics.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Plot Area
        self.plot_frame = ttk.Frame(self.dash_paned, style="Card.TFrame")
        self.dash_paned.add(self.plot_frame, weight=3)
        
        # This frame will hold the replay buttons, and is packed at the bottom.
        self.replay_controls = ttk.Frame(self.plot_frame, style="Card.TFrame")
        self.replay_controls.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        # This frame will hold the plot canvas, and is packed above the controls.
        canvas_container = ttk.Frame(self.plot_frame, style="Card.TFrame")
        canvas_container.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        # Initialize empty plot and add it to the canvas container
        self.fig = Figure(figsize=(8, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create replay buttons inside their dedicated frame. They will be hidden/shown as needed.
        self.btn_replay_play = ttk.Button(
            self.replay_controls, text="Play Stance Video", command=self.start_replay
        )
        self.btn_replay_pause = ttk.Button(
            self.replay_controls, text="Pause Video", command=self.stop_replay
        )
        # The buttons are not packed here; they are packed in _update_plots.


    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if path:
            self.current_csv_path = path
            self.lbl_filename.config(text=os.path.basename(path), foreground="black")
            
            # Auto-detect
            folder_name = os.path.basename(os.path.dirname(path))
            self.entry_participant.delete(0, tk.END)
            self.entry_participant.insert(0, folder_name)
            
            _, ex_key, variant = _infer_participant_and_exercise(os.path.dirname(path), os.path.basename(path))
            if ex_key:
                self._select_exercise_from_key(ex_key, variant)

    def _select_exercise_from_key(self, key, variant):
        mapping = {
            "comp_stepping_forward": "Compensatory Stepping (FORWARD)",
            "comp_stepping_backward": "Compensatory Stepping (BACKWARD)",
            "comp_stepping_lateral_left": "Compensatory Stepping (LATERAL-LEFT)",
            "comp_stepping_lateral_right": "Compensatory Stepping (LATERAL-RIGHT)",
            "rise_to_toes": "Rise to Toes",
            "sit_to_stand": "Sit to Stand",
            "stand_one_leg_left": "Stand on One Leg (Left)",
            "stand_one_leg_right": "Stand on One Leg (Right)",
            "stance_eyes_open": "Stance (Eyes Open)",
            "stance_eyes_closed": "Stance (Eyes Closed)",
        }
        
        if key in mapping:
            self.combo_exercise.set(mapping[key])
    
    def _on_exercise_change(self, event):
        self.lbl_score.config(text="-")
        self.lbl_score_desc.config(text="")

    def on_analyze(self):
        if not self.current_csv_path:
            messagebox.showwarning("Input Error", "Please select a CSV file first.")
            return
            
        ex_selection = self.combo_exercise.get()
        if not ex_selection:
            messagebox.showwarning("Input Error", "Please select an exercise type.")
            return
            
        participant = self.entry_participant.get().strip() or "Unknown"
        
        try:
            # Process based on selection
            if "Compensatory Stepping" in ex_selection:
                if "FORWARD" in ex_selection: direction = "FORWARD"
                elif "BACKWARD" in ex_selection: direction = "BACKWARD"
                elif "LATERAL-LEFT" in ex_selection: direction = "LATERAL-LEFT"
                elif "LATERAL-RIGHT" in ex_selection: direction = "LATERAL-RIGHT"
                else: direction = "UNKNOWN"
                
                self.current_result = process_compensatory_stepping(self.current_csv_path, direction, participant)
                
            elif "Rise to Toes" in ex_selection:
                self.current_result = process_rise_to_toes(self.current_csv_path, participant)
                
            elif "Sit to Stand" in ex_selection:
                self.current_result = process_sit_to_stand(self.current_csv_path, participant)
                
            elif "Stand on One Leg" in ex_selection:
                leg = "Left" if "Left" in ex_selection else "Right"
                self.current_result = process_stand_on_one_leg(self.current_csv_path, participant, leg)
                
            elif "Stance (Eyes Open)" in ex_selection:
                self.current_result = process_stance_eyes_open(self.current_csv_path, participant)
                
            elif "Stance (Eyes Closed)" in ex_selection:
                self.current_result = process_stance_eyes_closed(self.current_csv_path, participant)
            
            # Load signals for plotting
            self.current_signals = load_basic_signals(self.current_csv_path)
            
            # Update UI
            self._update_score_display()
            self._update_metrics_table()
            self._update_plots(ex_selection)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_score_display(self):
        if self.current_result:
            score = self.current_result.score
            self.lbl_score.config(text=str(score))
            
            if score == 2: desc = "Normal"
            elif score == 1: desc = "Moderate"
            else: desc = "Severe / Unable"
            self.lbl_score_desc.config(text=desc)

    def _update_metrics_table(self):
        for item in self.tree_metrics.get_children():
            self.tree_metrics.delete(item)
            
        if self.current_result:
            features = self.current_result.features
            for k, v in features.items():
                if k.startswith("_"): continue
                if isinstance(v, float):
                    val_str = f"{v:.3f}"
                else:
                    val_str = str(v)
                self.tree_metrics.insert("", "end", text=k, values=(val_str,))

    def _update_plots(self, ex_type):
        # Stop any running animation
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

        # Reset any running replay whenever plots are refreshed
        if self.replay_after_id is not None:
            try:
                self.after_cancel(self.replay_after_id)
            except Exception:
                pass
            self.replay_after_id = None
        self.replay_running = False
        self.replay_frames = None
        self.replay_indices = None
        self.replay_ax = None
        self.replay_im = None

        self.fig.clear()
        s = self.current_signals
        features = self.current_result.features
        
        # --- Hide/Show Replay Buttons Based on Exercise ---
        # First, always clear any existing buttons from the control frame
        for widget in self.replay_controls.winfo_children():
            widget.pack_forget()

        # Different layouts for different exercises
        if "Rise to Toes" in ex_type:
            self._plot_rise_to_toes(s, features)
        elif "Compensatory Stepping" in ex_type:
            self._plot_compensatory(s, features)
        elif "Sit to Stand" in ex_type:
            self._plot_sit_to_stand(s, features)
        elif "Stance (Eyes Open)" in ex_type or "Stance (Eyes Closed)" in ex_type:
            # For stance, pack the buttons into the now-visible controls frame
            self.btn_replay_play.pack(side=tk.LEFT, padx=(0, 5))
            self.btn_replay_pause.pack(side=tk.LEFT)
            self._plot_stance(s, features)
        else:
            self._plot_general_sway(s, features)
            
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_rise_to_toes(self, s, features):
        # 3 Plots: Heatmap, Contact Area, CoP Sway
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1])
        
        # 1. Average Pressure Map (Heatmap)
        ax0 = self.fig.add_subplot(gs[0])
        avg_frame = np.mean(s.frames, axis=0)
        im = ax0.imshow(avg_frame, cmap='hot', interpolation='nearest', aspect='auto')
        self.fig.colorbar(im, ax=ax0, label='Avg Pressure')
        ax0.set_title("Average Pressure Distribution (Heatmap)")
        
        # 2. Contact Area (Valid Frames Only)
        ax1 = self.fig.add_subplot(gs[1])
        
        valid_time = features.get("_valid_time")
        smoothed_area = features.get("_smoothed_area")
        
        if valid_time is not None and smoothed_area is not None and len(valid_time) == len(smoothed_area):
            # Sort by time just in case
            # But it should be sorted.
            ax1.plot(valid_time, smoothed_area, 'b-', label='Contact Area (Active Frames)')
        else:
             # Fallback to raw if processing failed
            ax1.plot(s.time_s, s.area, 'gray', alpha=0.5, label='Raw Area')
            
        baseline = features.get("Baseline Area (pixels)", 0)
        thresh = features.get("Area Threshold", 0)
        
        if baseline > 0:
            ax1.axhline(y=baseline, color='k', linestyle='--', label='Flat Foot Baseline')
            ax1.axhline(y=thresh, color='r', linestyle=':', label='Heel Rise Threshold (80%)')
            
        ax1.set_title("Contact Area Dynamics (Heel Rise Detection)")
        ax1.set_ylabel("Area (pixels)")
        ax1.set_xlabel("Time (s)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # 3. CoP Sway Velocity (Stability)
        ax2 = self.fig.add_subplot(gs[2])
        
        # Filter valid speed only (remove jumps due to gaps)
        dt = max(s.dt, 1e-4)
        vx = np.diff(s.cop_x) / dt
        vy = np.diff(s.cop_y) / dt
        speed = np.sqrt(vx**2 + vy**2)
        speed = np.concatenate(([0], speed))
        
        # Ensure time array matches speed array length
        time_array = s.time_s[:len(speed)] if len(s.time_s) > len(speed) else s.time_s
        
        # Mask out insane speeds (e.g. > 500 px/s) likely due to entry/exit or noise
        # Also mask out NaN
        valid_speed_mask = (speed < 500) & (~np.isnan(speed)) & (time_array < np.max(time_array) + 1)
        
        # Ensure mask is same length as both arrays
        if len(valid_speed_mask) == len(speed) == len(time_array):
            t_valid = time_array[valid_speed_mask]
            s_valid = speed[valid_speed_mask]
        else:
            # Fallback: use minimum length
            min_len = min(len(time_array), len(speed), len(valid_speed_mask))
            t_valid = time_array[:min_len][valid_speed_mask[:min_len]]
            s_valid = speed[:min_len][valid_speed_mask[:min_len]]
        
        if len(t_valid) > 2 and len(s_valid) > 2 and len(t_valid) == len(s_valid):
            # Smooth heavily for "Stability" trend (0.25s window)
            window_size = max(min(int(0.25 / dt), len(s_valid)), 3)
            # Use mode='same' to keep dimensions consistent with t_valid
            s_smooth = np.convolve(s_valid, np.ones(window_size)/window_size, mode='same')
            
            ax2.plot(t_valid, s_smooth, 'purple', linewidth=2, label='Stability (Smoothed)')
        elif len(t_valid) > 0 and len(s_valid) > 0:
            # If too few points for smoothing, just plot raw
            min_plot_len = min(len(t_valid), len(s_valid))
            ax2.plot(t_valid[:min_plot_len], s_valid[:min_plot_len], 'purple', linewidth=2, label='Stability')
            
            # Add markers for events if available
            if "Time on Toes (s)" in features:
                # We don't have exact start/end indices easily available here 
                # unless we passed them. 
                # But we can assume low speed = good stability.
                pass

        ax2.set_title("Postural Stability (CoP Speed)")
        ax2.set_ylabel("Sway Speed (px/s)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

    def _plot_compensatory(self, s, features):
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1])
        
        # 1. Heatmap (Max Pressure to see steps)
        ax0 = self.fig.add_subplot(gs[0])
        max_frame = np.max(s.frames, axis=0) # Max over time shows "footprints"
        im = ax0.imshow(max_frame, cmap='viridis', interpolation='nearest', aspect='auto')
        self.fig.colorbar(im, ax=ax0, label='Max Pressure')
        ax0.set_title("Max Pressure Map (Footprints Trace)")
        
        # 2. CoP Speed
        ax1 = self.fig.add_subplot(gs[1])
        dt = max(s.dt, 1e-4)
        speed = np.sqrt(np.diff(s.cop_x)**2 + np.diff(s.cop_y)**2) / dt
        ax1.plot(s.time_s[:-1], speed, 'b-')
        ax1.set_title("CoP Speed (Step Impulse)")
        ax1.set_ylabel("Speed")
        ax1.grid(True, alpha=0.3)
        
        # 3. CoP Trajectory
        ax2 = self.fig.add_subplot(gs[2])
        ax2.plot(s.cop_x, s.cop_y, 'k-', alpha=0.6)
        ax2.plot(s.cop_x[0], s.cop_y[0], 'go', label='Start')
        ax2.plot(s.cop_x[-1], s.cop_y[-1], 'rx', label='End')
        ax2.set_title("CoP Trajectory")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.axis('equal')
        ax2.legend()

    def _plot_sit_to_stand(self, s, features):
        gs = self.fig.add_gridspec(2, 1)
        
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(s.time_s, s.force, 'k-', label='Total Force')
        ax1.set_title("Total Force Profile (Loading)")
        ax1.set_ylabel("Force Sum")
        ax1.grid(True, alpha=0.3)
        
        max_f = features.get("Max Force", 0)
        if max_f > 0:
            ax1.axhline(y=max_f * 0.8, color='g', linestyle='--', label='Stability Threshold (80%)')
        ax1.legend()
        
        ax2 = self.fig.add_subplot(gs[1])
        # Plot CoP Sway only for the stable phase?
        # Or just general sway
        ax2.plot(s.cop_x, s.cop_y, 'purple', alpha=0.5)
        ax2.set_title("CoP Sway Pattern")
        ax2.axis('equal')

    def _plot_stance(self, s: BasicMatSignals, features):
        """
        Stance-specific visualization (eyes open / eyes closed):
        - Top-left: live replay of frames (used by the replay controls)
        - Top-right: average pressure map over the stance
        - Bottom: CoP trajectory on the mat
        """
        # 2x2 grid: replay | average heatmap
        #           CoP trajectory (full width)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

        # 1. Replay axis (this will be updated by the replay controls)
        ax_replay = self.fig.add_subplot(gs[0, 0])
        # Choose a sensible starting frame: first frame with meaningful force,
        # so that the video window is not just an empty black frame.
        if s.frames.size > 0:
            force = s.force
            area = s.area
            # Only use frames where we have meaningful contact, so feet are visible.
            # Criterion: total force above a very small threshold OR area > 0.
            active_mask = (force > 5.0) | (area > 0)
            active_idx = np.where(active_mask)[0]
            if active_idx.size > 0:
                # Use only active frames for the stance video
                self.replay_indices = active_idx.astype(int)
                self.replay_start_index = 0
                first_idx = self.replay_indices[0]
                first_frame = s.frames[first_idx]
                self.replay_time = s.time_s
                self.replay_start_time = float(s.time_s[first_idx])
            else:
                # Fallback: use all frames if we cannot detect active ones
                self.replay_indices = np.arange(len(s.frames), dtype=int)
                self.replay_start_index = 0
                first_idx = self.replay_indices[0]
                first_frame = s.frames[first_idx]
                self.replay_time = s.time_s
                self.replay_start_time = float(s.time_s[first_idx])
        else:
            self.replay_start_index = 0
            first_frame = np.zeros((10, 10))
            self.replay_time = None
            self.replay_start_time = 0.0
        im_replay = ax_replay.imshow(first_frame, cmap='hot', interpolation='nearest', aspect='auto')
        ax_replay.set_title("Pressure Map (Replay)")
        ax_replay.axis("off")

        # Store for replay controls
        self.replay_frames = s.frames
        self.replay_ax = ax_replay
        self.replay_im = im_replay

        # 2. Average pressure map
        ax_avg = self.fig.add_subplot(gs[0, 1])
        if s.frames.size > 0:
            avg_frame = np.mean(s.frames, axis=0)
        else:
            avg_frame = np.zeros((10, 10))
        im_avg = ax_avg.imshow(avg_frame, cmap='hot', interpolation='nearest', aspect='auto')
        self.fig.colorbar(im_avg, ax=ax_avg, fraction=0.046, pad=0.04)
        ax_avg.set_title("Average Pressure Distribution (Stance)")
        ax_avg.axis("off")

        # 3. CoP Sway over Time (Medial-Lateral and Anterior-Posterior)
        ax_sway_time = self.fig.add_subplot(gs[1, 0])
        
        # Use only valid CoP data
        valid_mask = np.isfinite(s.cop_x) & np.isfinite(s.cop_y)
        if np.any(valid_mask):
            time_valid = s.time_s[valid_mask]
            cop_x_valid = s.cop_x[valid_mask]
            cop_y_valid = s.cop_y[valid_mask]
            
            # De-mean to show sway around the center
            cop_x_demeaned = cop_x_valid - np.mean(cop_x_valid)
            cop_y_demeaned = cop_y_valid - np.mean(cop_y_valid)

            ax_sway_time.plot(time_valid, cop_x_demeaned, 'r-', alpha=0.8, label='M/L Sway (X)')
            ax_sway_time.plot(time_valid, cop_y_demeaned, 'b-', alpha=0.8, label='A/P Sway (Y)')
        
        ax_sway_time.set_title("CoP Sway Over Time")
        ax_sway_time.set_xlabel("Time (s)")
        ax_sway_time.set_ylabel("Sway (sensor units)")
        ax_sway_time.legend(loc="best")
        ax_sway_time.grid(True, alpha=0.3)

        # 4. Sway Velocity over Time (Stability)
        ax_sway_vel = self.fig.add_subplot(gs[1, 1])
        
        dt = max(s.dt, 1e-4)
        vx = np.diff(s.cop_x) / dt
        vy = np.diff(s.cop_y) / dt
        speed = np.sqrt(vx**2 + vy**2)
        
        # Use time from the midpoint of diff
        time_for_speed = s.time_s[:-1]
        
        # Filter out extreme speeds from noise/gaps
        valid_speed_mask = (speed < 500) & np.isfinite(speed)
        
        if np.any(valid_speed_mask):
            ax_sway_vel.plot(time_for_speed[valid_speed_mask], speed[valid_speed_mask], 'purple', alpha=0.8, label='Sway Speed')

        ax_sway_vel.set_title("Postural Stability (Sway Speed)")
        ax_sway_vel.set_xlabel("Time (s)")
        ax_sway_vel.set_ylabel("Sway Speed (units/s)")
        ax_sway_vel.legend(loc="best")
        ax_sway_vel.grid(True, alpha=0.3)

    def _plot_general_sway(self, s, features):
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
        
        # Heatmap
        ax0 = self.fig.add_subplot(gs[0])
        avg_frame = np.mean(s.frames, axis=0)
        im = ax0.imshow(avg_frame, cmap='hot', interpolation='nearest', aspect='auto')
        self.fig.colorbar(im, ax=ax0)
        ax0.set_title("Average Pressure Distribution")
        
        # Sway
        ax1 = self.fig.add_subplot(gs[1])
        ax1.plot(s.cop_x, s.cop_y, 'k-', alpha=0.6)
        ax1.set_title("Center of Pressure Trajectory")
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

    def _update_animation_frame(self, i):
        """Update function for FuncAnimation."""
        # i is the frame number from 0 to len(self.replay_indices)-1
        if not hasattr(self, 'replay_indices') or self.replay_indices is None or i >= len(self.replay_indices):
            return

        frame_idx = self.replay_indices[i]
        frame = self.replay_frames[frame_idx]
        self.replay_im.set_data(frame)

        # Update title with time
        if self.replay_time is not None:
            t_abs = float(self.replay_time[frame_idx])
            t_rel = max(0.0, t_abs - float(self.replay_start_time))
            self.replay_ax.set_title(f"Pressure Map (Replay)  t = {t_rel:.2f} s")

        # When blit=False, we don't return artists. The function just modifies the plot.
        return

    def _on_animation_end(self):
        """Callback for when the animation finishes."""
        # Set animation to None so we know it has finished.
        self.animation = None

    def start_replay(self):
        """Start replaying frames over time using FuncAnimation."""
        if self.replay_frames is None or self.replay_ax is None or not hasattr(self, 'replay_indices') or self.replay_indices is None:
            return

        # Stop any previous animation that might be running
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

        # Calculate the average interval for a smooth playback that matches total duration
        if len(self.replay_indices) > 1 and self.replay_time is not None:
            first_time = self.replay_time[self.replay_indices[0]]
            last_time = self.replay_time[self.replay_indices[-1]]
            total_duration_s = last_time - first_time
            if total_duration_s > 0:
                avg_interval_ms = (total_duration_s * 1000) / len(self.replay_indices)
            else:
                avg_interval_ms = 40 # 25 FPS
        else:
            avg_interval_ms = 40 # Fallback to 25 FPS

        # Create the animation. blit=False is more robust for GUIs with text updates in TkAgg
        self.animation = FuncAnimation(
            self.fig,
            self._update_animation_frame,
            frames=len(self.replay_indices),
            interval=max(1, avg_interval_ms),
            blit=False, # blit=False is crucial for stability with text updates in TkAgg
            repeat=False,
            # save_count is needed to ensure the _on_animation_end callback is triggered
            save_count=len(self.replay_indices)
        )
        # Connect the end-of-animation callback
        self.animation._stop = self._on_animation_end
        self.canvas.draw()


    def stop_replay(self):
        """Pause the frame replay."""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None

if __name__ == "__main__":
    app = ModernMiniBESTGui()
    app.mainloop()
