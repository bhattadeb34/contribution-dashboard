# generalized_contribution_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, LabelSet
from bokeh.transform import dodge
import io

# =============================================================================
# GENERALIZED CONTRIBUTION ANALYSIS DASHBOARD
# =============================================================================

def calculate_proportional_strategy(df, target_amount, base_column='current_contribution', 
                                   exclude_statuses=['blocked', 'excluded']):
    """Calculate proportional contribution increases to reach target."""
    active_df = df[~df['status'].isin(exclude_statuses)].copy()
    
    if active_df.empty:
        return df
    
    total_current = active_df[base_column].sum()
    
    if total_current == 0:
        return df
    
    # Calculate proportional increases
    df_result = df.copy()
    
    for idx, row in active_df.iterrows():
        proportion = row[base_column] / total_current
        required_increase = target_amount * proportion
        new_total = row[base_column] + required_increase
        
        # Apply capacity constraint if exists
        if pd.notna(row['max_capacity']) and new_total > row['max_capacity']:
            new_total = row['max_capacity']
        
        df_result.loc[idx, 'suggested_contribution'] = new_total
    
    return df_result

def calculate_equal_strategy(df, target_amount, base_column='current_contribution',
                           exclude_statuses=['blocked', 'excluded']):
    """Calculate equal contribution increases to reach target."""
    active_df = df[~df['status'].isin(exclude_statuses)].copy()
    
    if active_df.empty:
        return df
    
    equal_increase = target_amount / len(active_df)
    df_result = df.copy()
    
    for idx, row in active_df.iterrows():
        new_total = row[base_column] + equal_increase
        
        # Apply capacity constraint if exists
        if pd.notna(row['max_capacity']) and new_total > row['max_capacity']:
            new_total = row['max_capacity']
        
        df_result.loc[idx, 'suggested_contribution'] = new_total
    
    return df_result

def calculate_capacity_based_strategy(df, target_amount, base_column='current_contribution',
                                    exclude_statuses=['blocked', 'excluded']):
    """Calculate capacity-based contribution increases."""
    active_df = df[~df['status'].isin(exclude_statuses)].copy()
    
    if active_df.empty:
        return df
    
    df_result = df.copy()
    remaining_target = target_amount
    
    # Sort by capacity utilization potential
    active_df['capacity_potential'] = np.where(
        pd.notna(active_df['max_capacity']),
        active_df['max_capacity'] - active_df[base_column],
        active_df[base_column] * 0.5  # 50% increase if no capacity limit
    )
    
    active_df = active_df.sort_values('capacity_potential', ascending=False)
    
    for idx, row in active_df.iterrows():
        if remaining_target <= 0:
            break
        
        max_increase = row['capacity_potential']
        actual_increase = min(remaining_target, max_increase)
        new_total = row[base_column] + actual_increase
        
        df_result.loc[idx, 'suggested_contribution'] = new_total
        remaining_target -= actual_increase
    
    return df_result

def create_comparison_chart(df, name_col, base_col, suggested_col):
    """Create interactive comparison chart."""
    # Separate active and inactive contributors
    active_df = df[df['status'].isin(['active', 'conditional'])].copy()
    inactive_df = df[~df['status'].isin(['active', 'conditional'])].copy()
    
    p = figure(
        x_range=list(df[name_col]),
        height=600,
        title="Contribution Analysis Dashboard",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Tooltips
    TOOLTIPS = """
        <div style="padding: 8px; border: 2px solid #333; border-radius: 8px; background-color: white;">
            <div style="font-size: 14px; font-weight: bold;">@name</div>
            <div style="font-size: 12px; margin: 4px 0;"><strong>Status:</strong> @status</div>
            <div style="font-size: 12px; margin: 4px 0;"><strong>Current:</strong> @current</div>
            <div style="font-size: 12px; margin: 4px 0;"><strong>Suggested:</strong> @suggested</div>
            <div style="font-size: 12px; margin: 4px 0;"><strong>Increase:</strong> @increase</div>
        </div>
    """
    
    p.add_tools(HoverTool(tooltips=TOOLTIPS))
    
    # Active contributors - side by side bars
    if not active_df.empty:
        current_source = ColumnDataSource({
            'name': active_df[name_col],
            'current': active_df[base_col],
            'suggested': active_df[suggested_col],
            'increase': active_df[suggested_col] - active_df[base_col],
            'status': active_df['status']
        })
        
        suggested_source = ColumnDataSource({
            'name': active_df[name_col],
            'current': active_df[base_col],
            'suggested': active_df[suggested_col],
            'increase': active_df[suggested_col] - active_df[base_col],
            'status': active_df['status']
        })
        
        # Current contributions (blue)
        p.vbar(x=dodge('name', -0.15, range=p.x_range), top='current', width=0.25,
               source=current_source, color="#3498DB", alpha=0.8, legend_label="Current")
        
        # Suggested contributions (green)
        p.vbar(x=dodge('name', 0.15, range=p.x_range), top='suggested', width=0.25,
               source=suggested_source, color="#27AE60", alpha=0.8, legend_label="Suggested")
    
    # Inactive contributors - single bars
    if not inactive_df.empty:
        inactive_df_copy = inactive_df.copy()
        inactive_df_copy['status_label'] = inactive_df_copy['status'].str.title()
        
        inactive_source = ColumnDataSource({
            'name': inactive_df_copy[name_col],
            'current': inactive_df_copy[base_col],
            'suggested': inactive_df_copy[base_col],  # Same as current
            'increase': [0] * len(inactive_df_copy),
            'status': inactive_df_copy['status']
        })
        
        p.vbar(x='name', top='current', width=0.4, source=inactive_source,
               color="#95A5A6", alpha=0.8)
        
        # Add status labels
        labels = LabelSet(
            x='name', y='current', text='status',
            source=inactive_source, text_align='center',
            text_baseline='bottom', y_offset=5, angle=np.pi/2,
            text_font_size="8pt", text_color="gray"
        )
        p.add_layout(labels)
    
    # Styling
    p.x_range.range_padding = 0.05
    p.xaxis.major_label_orientation = np.pi / 4
    p.yaxis.formatter = NumeralTickFormatter(format="₹0,0")
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    
    return p

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(layout="wide", page_title="Contribution Analysis Dashboard")

st.title("Contribution Analysis Dashboard")
st.markdown("Upload your contributor data and analyze funding strategies for any goal")

# File upload
uploaded_file = st.file_uploader(
    "Upload Contributor Data (CSV)", 
    type="csv",
    help="Upload a CSV file with contributor information"
)

# Sample data generation button
if st.button("Generate Sample Data"):
    sample_data = """name,current_contribution,max_capacity,status,notes
Manab,26000,40000,active,High capacity contributor
Papia,20000,20000,blocked,Deceased - fixed deposit
Tinku,16000,30000,active,High capacity contributor  
Krishna,15500,20000,active,Key contributor
Purnima,15500,20000,active,Key contributor
Piu,15700,22000,active,Key contributor
Bharati,13000,17000,active,Key contributor
Sujash,12000,15000,conditional,Can help if needed
Rama,11000,15000,active,Key contributor
Bui,10000,14000,active,Regular contributor
Rekha,10000,14000,active,Regular contributor
Shukla,6000,9000,conditional,Limited capacity
Soma,6500,9500,conditional,Limited capacity
Shyamal,20000,20000,blocked,Personal constraints
Papri_Mahua,5500,5500,blocked,Cannot increase
Santu,5000,5000,blocked,Cannot increase
Swarup,3000,3000,excluded,Too small
Arup,3000,3000,excluded,Too small"""
    
    st.download_button(
        label="Download Sample CSV",
        data=sample_data,
        file_name="sample_contributors.csv",
        mime="text/csv"
    )

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['name', 'current_contribution', 'status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Required columns: name, current_contribution, status")
            st.info("Optional columns: max_capacity, notes")
            st.stop()
        
        # Add missing optional columns
        if 'max_capacity' not in df.columns:
            df['max_capacity'] = np.nan
        if 'notes' not in df.columns:
            df['notes'] = ""
        
        # Initialize suggested contribution column
        df['suggested_contribution'] = df['current_contribution'].copy()
        
        # Sidebar configuration
        st.sidebar.header("Goal Configuration")
        
        # Goal settings
        goal_name = st.sidebar.text_input("Goal Name", value="Full Catering")
        target_amount = st.sidebar.number_input(
            "Target Amount Needed", 
            min_value=0, 
            value=56300,
            step=1000,
            help="Total additional amount needed to reach the goal"
        )
        
        current_total = df['current_contribution'].sum()
        st.sidebar.metric("Current Total Contributions", f"₹{current_total:,}")
        st.sidebar.metric("Additional Amount Needed", f"₹{target_amount:,}")
        
        # Strategy selection
        strategy = st.sidebar.selectbox(
            "Distribution Strategy",
            ["Manual", "Proportional", "Equal Increase", "Capacity-Based"],
            help="How to distribute the target amount among contributors"
        )
        
        # Apply strategy
        if strategy == "Proportional":
            df = calculate_proportional_strategy(df, target_amount)
        elif strategy == "Equal Increase":
            df = calculate_equal_strategy(df, target_amount)
        elif strategy == "Capacity-Based":
            df = calculate_capacity_based_strategy(df, target_amount)
        
        # Manual adjustment controls
        if strategy == "Manual" or st.sidebar.checkbox("Enable Manual Adjustments"):
            st.sidebar.subheader("Manual Adjustments")
            
            active_contributors = df[~df['status'].isin(['blocked', 'excluded'])]
            
            for idx, row in active_contributors.iterrows():
                current_suggested = df.loc[idx, 'suggested_contribution']
                max_val = row['max_capacity'] if pd.notna(row['max_capacity']) else current_suggested * 2
                
                new_value = st.sidebar.number_input(
                    f"{row['name']}",
                    min_value=float(row['current_contribution']),
                    max_value=float(max_val),
                    value=float(current_suggested),
                    step=500.0,
                    key=f"manual_{idx}"
                )
                
                df.loc[idx, 'suggested_contribution'] = new_value
        
        # Calculate metrics
        df['increase'] = df['suggested_contribution'] - df['current_contribution']
        df['increase_pct'] = np.where(
            df['current_contribution'] > 0,
            (df['increase'] / df['current_contribution'] * 100).round(1),
            0
        )
        
        total_generated = df['increase'].sum()
        success_rate = (total_generated / target_amount * 100) if target_amount > 0 else 100
        
        # Main dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Target Goal", f"₹{target_amount:,}")
        with col2:
            st.metric("Strategy Generates", f"₹{total_generated:,}", 
                     delta=f"₹{total_generated - target_amount:,}")
        with col3:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Progress bar
        st.progress(min(1.0, total_generated / target_amount) if target_amount > 0 else 1.0)
        
        # Interactive chart
        st.subheader(f"Contribution Analysis for: {goal_name}")
        chart = create_comparison_chart(df, 'name', 'current_contribution', 'suggested_contribution')
        st.bokeh_chart(chart, use_container_width=True)
        
        # Summary by status
        st.subheader("Strategy Summary")
        summary_df = df.groupby('status').agg({
            'name': 'count',
            'current_contribution': 'sum',
            'increase': 'sum',
            'increase_pct': 'mean'
        }).rename(columns={
            'name': 'Contributors',
            'current_contribution': 'Current Total',
            'increase': 'Total Increase',
            'increase_pct': 'Avg Increase %'
        })
        
        summary_df = summary_df[summary_df['Contributors'] > 0]
        st.dataframe(summary_df.style.format({
            'Current Total': '₹{:,.0f}',
            'Total Increase': '₹{:,.0f}',
            'Avg Increase %': '{:.1f}%'
        }), use_container_width=True)
        
        # Detailed breakdown
        st.subheader("Detailed Breakdown")
        display_df = df[['name', 'current_contribution', 'suggested_contribution', 
                        'increase', 'increase_pct', 'status', 'notes']].copy()
        
        display_df = display_df.rename(columns={
            'name': 'Name',
            'current_contribution': 'Current',
            'suggested_contribution': 'Suggested',
            'increase': 'Increase',
            'increase_pct': 'Increase %',
            'status': 'Status',
            'notes': 'Notes'
        })
        
        st.dataframe(display_df.style.format({
            'Current': '₹{:,.0f}',
            'Suggested': '₹{:,.0f}',
            'Increase': '₹{:,.0f}',
            'Increase %': '{:.1f}%'
        }), use_container_width=True)
        
        # Export functionality
        if st.button("Export Strategy"):
            export_df = df[['name', 'current_contribution', 'suggested_contribution', 'increase']].copy()
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Strategy CSV",
                data=csv,
                file_name=f"{goal_name.lower().replace(' ', '_')}_strategy.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please check your CSV format and try again")

else:
    st.info("Upload a CSV file to get started, or click 'Generate Sample Data' to see the required format")
    
    # Show required format
    st.subheader("Required CSV Format")
    st.code("""
name,current_contribution,max_capacity,status,notes
John Doe,10000,15000,active,Regular contributor
Jane Smith,5000,5000,blocked,Cannot increase
Bob Johnson,8000,,conditional,Can help if needed
    """)
    
    st.markdown("""
    **Column Descriptions:**
    - `name`: Contributor name (required)
    - `current_contribution`: Current contribution amount (required)
    - `max_capacity`: Maximum they can contribute (optional)
    - `status`: active/conditional/blocked/excluded (required)
    - `notes`: Additional information (optional)
    """)