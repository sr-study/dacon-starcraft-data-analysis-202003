from sklearn.cluster import KMeans

def cluster_with_features(df):
    kmeans = KMeans(n_clusters=50)
    df = df.fillna(0)
    y = kmeans.fit_predict(df[['time', 'p1_camera_x_var', 'p1_camera_y_var', 'p0_camera_x_var', 'p0_camera_y_var', 'p0_event_GetControlGroup', 'p1_event_GetControlGroup', 'p0_event_Selection', 'p0_event_Ability','p1_event_Ability','p1_ability_(5A0) - Attack', 'p0_ability_(5A0) - Attack']])
    df['Cluster'] = y

    return df[['Cluster']]
