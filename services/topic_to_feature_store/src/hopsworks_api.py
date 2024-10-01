from typing import List

import pandas as pd
import hopsworks

from src.config import hopsworks_config as config

def push_value_to_feature_group(
    value: dict, 
    feature_group_name: str, 
    feature_group_version: int,
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool,
):
    """
    Pushes the given "value" to the given "feature_group_name" in the feature store.
    
    Args:
        value (dict): The value to push to the feature store.
        feature_group_name (str): The name of the feature group to push the value to.
        feature_group_version (int): The version of the feature group to push the value to.
        feature_group_primary_key (List[str]): The primary key of the feature group to push the value to.
        feature_group_event_time (str): The event time of the feature group to push the value to.
        start_offline_materialization(bool): Whether to start the offline materialization or not when we 
        save the 'value' to the feature group
    Returns:
        None
    """
    
    project = hopsworks.login(
        project=config.hopsworks_project_name,
        api_key_value=config.hopsworks_api_key,
    )
    
    feature_store = project.get_feature_store()
    
    # Get or create feature group
    feature_group = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        primary_key=feature_group_primary_keys,
        event_time=feature_group_event_time,
        online_enabled=True,
        # expectation_suite_name="expectation_suite_name"
    )
    
    # Transform the value to a pandas dataframe
    value_df = pd.DataFrame([value])
    # breakpoint()
    
    # Insert data into feature group
    feature_group.insert(value_df, 
                         write_options={"start_offline_materialization": start_offline_materialization})