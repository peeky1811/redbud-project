
"""MLayalee: Additional EDA visualizations."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parent

# Load the cleaned dataset.
data = pd.read_csv(ROOT / 'redbud_cleaned_candidates.csv')

print('Dataset shape:', data.shape)

# Count observations for each class in every month.
monthly = pd.crosstab(
    data['month'],
    data['flowering']
).reindex(range(1, 13), fill_value=0)

monthly = monthly.reindex(columns=[0, 1], fill_value=0)

monthly.columns = ['no_flowers_fruits', 'flowering']

# Calculate total observations and flowering proportion.
monthly['total'] = (
    monthly['no_flowers_fruits'] + monthly['flowering']
)

monthly['flowering_proportion'] = (
    monthly['flowering'] / monthly['total']
)

print('\nMonthly flowering proportions:')
print(monthly)

# Create the plot.
fig, ax = plt.subplots(figsize=(10, 5))

months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

ax.bar(
    months,
    monthly['flowering_proportion'] * 100,
    color='mediumseagreen',
    edgecolor='black',
    linewidth=0.5
)

ax.set_xlabel('Observation month')
ax.set_ylabel('Flowering observations (%)')
ax.set_title('Monthly Flowering Proportion of Eastern Redbud in Texas')
ax.set_ylim(0, 105)

ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    ROOT / 'monthly_flowering_proportion.png',
    dpi=200
)

plt.close()

print('\nSaved monthly_flowering_proportion.png')

# Geographic distribution of annotated observations.
fig, ax = plt.subplots(figsize=(9, 7))

nonflowering = data[data['flowering'] == 0]
flowering = data[data['flowering'] == 1]

ax.scatter(
    nonflowering['longitude'],
    nonflowering['latitude'],
    c='steelblue',
    label='No flowers/fruits',
    s=18,
    alpha=0.6
)

ax.scatter(
    flowering['longitude'],
    flowering['latitude'],
    c='deeppink',
    label='Flowering',
    s=18,
    alpha=0.6
)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(
    'Geographic Distribution of Eastern Redbud Observations in Texas'
)

ax.set_xlim(-107, -93)
ax.set_ylim(25, 37)

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    ROOT / 'texas_redbud_distribution.png',
    dpi=200
)

plt.close()

print('Saved texas_redbud_distribution.png')

# Improved geographic map with Texas boundary.

states_url = (
    'https://raw.githubusercontent.com/PublicaMundi/'
    'MappingAPI/master/data/geojson/us-states.json'
)

states = gpd.read_file(states_url)
texas = states[states['name'] == 'Texas']

if texas.empty:
    raise ValueError('Texas boundary not found in GeoJSON.')

fig, ax = plt.subplots(figsize=(10, 8))

# Draw Texas boundary.
texas.plot(
    ax=ax,
    facecolor='#f4f4f4',
    edgecolor='black',
    linewidth=1.2
)

# Draw observation locations.
nonflowering = data[data['flowering'] == 0]
flowering = data[data['flowering'] == 1]

ax.scatter(
    nonflowering['longitude'],
    nonflowering['latitude'],
    color='steelblue',
    label='No flowers/fruits',
    s=18,
    alpha=0.6,
    zorder=2
)

ax.scatter(
    flowering['longitude'],
    flowering['latitude'],
    color='deeppink',
    label='Flowering',
    s=18,
    alpha=0.6,
    zorder=3
)

# Add major Texas city labels for geographic reference.
# Coordinates are approximate city-center locations.
cities = {
    'Dallas': (-96.7970, 32.7767),
    'Fort Worth': (-97.3308, 32.7555),
    'Austin': (-97.7431, 30.2672),
    'Houston': (-95.3698, 29.7604),
    'San Antonio': (-98.4936, 29.4241)
}

# Adjust label positions to avoid covering observation clusters.
label_offsets = {
    'Dallas': (8, 12),
    'Fort Worth': (-8, -16),
    'Austin': (8, 10),
    'Houston': (8, -23),
    'San Antonio': (-8, -16)
}

for city, (lon, lat) in cities.items():

    # Mark the city center.
    ax.scatter(
        lon,
        lat,
        marker='*',
        s=75,
        color='black',
        edgecolor='white',
        linewidth=0.6,
        zorder=5
    )

    # Add the city name near its marker.
    ax.annotate(
        city,
        xy=(lon, lat),
        xytext=label_offsets[city],
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color='black',
        ha='left' if label_offsets[city][0] > 0 else 'right',
        va='center',
        bbox=dict(
            facecolor='white',
            alpha=0.75,
            edgecolor='none',
            pad=1.5
        ),
        zorder=6
    )

ax.set_title(
    'Eastern Redbud Phenology Observations in Texas'
)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

ax.set_xlim(-107, -93)
ax.set_ylim(25, 37)

# Correct for longitude distortion at Texas latitudes.
ax.set_aspect(1 / np.cos(np.deg2rad(31)))

ax.legend(loc='upper right')
ax.grid(alpha=0.15)

plt.tight_layout()

plt.savefig(
    ROOT / 'texas_redbud_distribution_improved.png',
    dpi=200
)

plt.close()

print('Saved texas_redbud_distribution_improved.png')
