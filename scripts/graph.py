import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import webbrowser
import os

def bar(df, x, hue, col, value_column=None, show_as_percent=False, orientation='vertical', col_order=None, hue_order=None, title=None):
    # Prepare the data
    if value_column:
        # If a value_column is provided, sum values by the group
        df_grouped = df.groupby([x, hue, col])[value_column].sum().reset_index()
    else:
        # If no value_column, count the occurrences
        df_grouped = df.groupby([x, hue, col]).size().reset_index(name='value')

    # If showing as percentage, calculate percentage
    if show_as_percent:
        total_values = df_grouped.groupby([col])[value_column if value_column else 'value'].sum().reset_index(name='total')
        df_grouped = pd.merge(df_grouped, total_values, on=col)
        df_grouped['percentage'] = df_grouped[value_column if value_column else 'value'] / df_grouped['total']
        y_column = 'percentage'
    else:
        y_column = value_column if value_column else 'value'

    # Set the style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("husl", len(hue_order) if hue_order else df[hue].nunique())

    # Create the plot
    if orientation == 'vertical':
        g = sns.catplot(
            data=df_grouped, kind="bar",
            x=x, y=y_column, hue=hue,
            col=col, col_order=col_order,
            hue_order=hue_order, palette=palette,
            height=6, aspect=1.2
        )
    else:
        g = sns.catplot(
            data=df_grouped, kind="bar",
            x=y_column, y=x, hue=hue,
            col=col, col_order=col_order,
            hue_order=hue_order, palette=palette,
            height=6, aspect=1.2
        )

    # Adjust the title and axis labels
    if title:
        g.fig.suptitle(title, y=1.02)

    g.set_axis_labels(
        "% of Total Count" if show_as_percent and orientation == 'horizontal' else "", 
        "% of Total Count" if show_as_percent and orientation == 'vertical' else "Count"
    )
    g.set_titles("{col_name}")
    g.despine(left=True)
    
    if show_as_percent:
        if orientation == 'vertical':
            g.set(ylim=(0, 1))
        else:
            g.set(xlim=(0, 1))

    # Add labels to each bar
    for ax in g.axes.flat:
        for p in ax.patches:
            if orientation == 'vertical':
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2., height,
                    f'{height*100:.1f}%' if show_as_percent else f'{int(height)}',
                    ha="center", va="bottom"
                )
            else:
                width = p.get_width()
                ax.text(
                    width, p.get_y() + p.get_height() / 2.,
                    f'{width*100:.1f}%' if show_as_percent else f'{int(width)}',
                    ha="left", va="center"
                )

    plt.show()



def sankey(df_input, node_columns, value_column=None, width=800, height=600, title=None, show_as_percent=False, label_font_size=10, label_font_color='black', show_labels=True, node_colors=None):
    try:
        df = df_input.copy()
        
        # Create the node dataframe
        node = pd.DataFrame()

        for col in node_columns:
            df[col] = df[col].apply(lambda x: f'{col}-{x}' if pd.notna(x) else None)
            if value_column is not None:
                node_sub = df.groupby(col)[value_column].sum().reset_index()
            else:
                node_sub = df.groupby(col).size().reset_index(name='value')
            node_sub.columns = ['node', 'value']
            node = pd.concat([node, node_sub], axis=0)

        # Remove duplicate nodes and create IDs
        node = node.groupby('node')['value'].sum().reset_index()
        node['id'] = range(len(node))

        # Assign colors to nodes if provided
        if node_colors:
            color_list = [node_colors.get(label, 'blue') for label in node['node']]
        else:
            color_list = 'blue'  # Default color

        # Convert value to millions if value_column is provided
        if value_column is not None:
            node['value'] = node['value'] / 1000000
            node['value'] = node['value'].apply(lambda x: f'{int(x)}M')

        # Create the sankey dataframe
        sankey = pd.DataFrame()

        for col, next_col in zip(node_columns[:-1], node_columns[1:]):
            if value_column is not None:
                sankey_sub = df.groupby([col, next_col])[value_column].sum().reset_index()
            else:
                sankey_sub = df.groupby([col, next_col]).size().reset_index(name='value')
            sankey_sub.columns = ['source', 'target', 'value']
            sankey = pd.concat([sankey, sankey_sub], axis=0)

        # If showing as percent, calculate percentages
        if show_as_percent and value_column is not None:
            total_value = sankey['value'].sum()
            sankey['value'] = (sankey['value'] / total_value) * 100
            sankey['value'] = sankey['value'].apply(lambda x: f'{x:.2f}%')

        # Create links
        sankey = pd.merge(sankey, node[['node', 'id']], how='left', left_on='source', right_on='node')
        sankey.rename(columns={'id': 'source_id'}, inplace=True)
        sankey = pd.merge(sankey, node[['node', 'id']], how='left', left_on='target', right_on='node')
        sankey.rename(columns={'id': 'target_id'}, inplace=True)

        sankey.drop(['node_x', 'node_y'], axis=1, inplace=True)

        # Blend edges to match the node colors
        link_colors = ['rgba(0, 0, 0, 0.5)' if node_colors is None else 
                       f'rgba({node_colors[source].replace("rgb(", "").replace(")", "")},0.5)' 
                       for source in sankey['source']]

        # Prepare the label configuration
        label_config = dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node['node'] if show_labels else None,
            color=color_list
        )

        # Build the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=label_config,
            link=dict(
                source=sankey['source_id'],
                target=sankey['target_id'],
                value=sankey['value'],
                color=link_colors
            )
        )])

        # Update layout
        fig.update_layout(
            title_text=title if title else 'Sankey Diagram',
            width=width,
            height=height
        )
        
        # Ensure output directory exists
        os.makedirs('output/graph', exist_ok=True)

        # Save as HTML and open in browser
        fig_path = os.path.abspath(f'output/graph/{title}.html')
        fig.write_html(fig_path)
        webbrowser.open(fig_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception to allow further handling if needed

# Example usage:
# sankey(df_input, ['Column1', 'Column2', 'Column3'], value_column='ValueColumn', show_as_percent=True, show_labels=False, node_colors={'Column1-A': 'rgb(255,0,0)', 'Column2-B': 'rgb(0,255,0)'})
