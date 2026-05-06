import numpy as np

def split_iemocap(df):
    """Partition dataset into five distinct cross validation folds based on speaker session identifiers"""
    # Extract session prefix like Ses01 from filename
    df['session'] = df['FileName'].apply(lambda x: x[:5])
    sessions = sorted(df['session'].unique())
    # Ensure exactly five sessions are present in dataset
    assert len(sessions) == 5, f"Expected 5 sessions but found {len(sessions)} sessions"
    # Store structural splits for five folds
    folds = []
    # Create fold allocations per session
    for test_session in sessions:
        # Assign current session samples to testing set
        test_idx = df[df['session'] == test_session].index.values
        # Assign remaining session samples to potential training set
        other_sessions_idx = df[df['session'] != test_session].index.values
        # Randomize order of remaining samples
        np.random.shuffle(other_sessions_idx)
        # Calculate validation set size as twenty percent of remaining samples
        eval_size = int(len(other_sessions_idx) * 0.2)
        # Separate randomized indices into validation and training sets
        eval_idx = other_sessions_idx[:eval_size]
        train_idx = other_sessions_idx[eval_size:]
        # Persist fold partitions into dict structure
        fold_info = {
            'train_idx': train_idx,
            'eval_idx': eval_idx,
            'test_idx': test_idx
        }
        folds.append(fold_info)
        # Log granular set counts for current fold
        print(f"\nFold for test session {test_session}:")
        print(f"Training set: {len(train_idx)} samples")
        print(f"Validation set: {len(eval_idx)} samples")
        print(f"Test set: {len(test_idx)} samples")
        # Extract unique sessions participating across different sets
        train_sessions = sorted(df.iloc[train_idx]['session'].unique())
        eval_sessions = sorted(df.iloc[eval_idx]['session'].unique())
        test_sessions = sorted(df.iloc[test_idx]['session'].unique())
    
    return folds

def split_msppodcast(df):
    """Directly partition dataset using predefined Split_Set feature column"""
    # Extract indices based on defined split indicators
    train_idx = df[df['Split_Set'] == 'Train'].index.values
    eval_idx = df[df['Split_Set'] == 'Development'].index.values
    test_idx = df[df['Split_Set'] == 'Test1'].index.values
    
    # Calculate unique speaker counts per partition
    train_speakers = df[df['Split_Set'] == 'Train']['SpkrID'].nunique()
    eval_speakers = df[df['Split_Set'] == 'Development']['SpkrID'].nunique()
    test_speakers = df[df['Split_Set'] == 'Test1']['SpkrID'].nunique()
    
    # Output detailed statistical breakdowns
    print("\nDataset Split Statistics:")
    print(f"Training set: {len(train_idx)} samples with {train_speakers} speakers")
    print(f"Development set: {len(eval_idx)} samples with {eval_speakers} speakers")
    print(f"Test1 set: {len(test_idx)} samples with {test_speakers} speakers")
    
    # Compute proportional dataset allocation
    total_samples = len(train_idx) + len(eval_idx) + len(test_idx)
    print(f"\nActual split ratio:")
    print(f"Train: {len(train_idx)/total_samples:.1%}")
    print(f"Development: {len(eval_idx)/total_samples:.1%}")
    print(f"Test1: {len(test_idx)/total_samples:.1%}")
    
    return [{
        'train_idx': train_idx,
        'eval_idx': eval_idx,
        'test_idx': test_idx
    }]