import os
import csv
import pandas as pd
import numpy as np
from sqlmodel import Session, select, delete
from app.models.models import ProjectInfo, Module, Feature, CodeMap, GraphEdge
from app.db.session import engine


def clear_project_data(session: Session, project_id: int):
    """
    Clear all related table data for a specified project_id.
    Deletion order: Dependent tables -> Dependency tables
    """
    # 1. Find all modules for this project
    statement_modules = select(Module.id).where(Module.repo == project_id)
    module_ids = session.exec(statement_modules).all()

    if module_ids:
        # 2. Find all features for these modules
        statement_features = select(Feature.id).where(Feature.module.in_(module_ids))
        feature_ids = session.exec(statement_features).all()

        if feature_ids:
            # 3. Delete CodeMaps (depend on Features)
            statement_delete_codemap = delete(CodeMap).where(CodeMap.feature.in_(feature_ids))
            session.exec(statement_delete_codemap)

            # 4. Delete Features (depend on Modules)
            statement_delete_feature = delete(Feature).where(Feature.id.in_(feature_ids))
            session.exec(statement_delete_feature)

        # 5. Delete Modules (depend on ProjectInfo)
        statement_delete_module = delete(Module).where(Module.id.in_(module_ids))
        session.exec(statement_delete_module)

    # 6. Delete GraphEdges (depend on ProjectInfo)
    statement_delete_edge = delete(GraphEdge).where(GraphEdge.repo == project_id)
    session.exec(statement_delete_edge)

    # 7. Reset summary_flag in ProjectInfo
    project = session.get(ProjectInfo, project_id)
    if project:
        project.summary_flag = False
        session.add(project)

    session.commit()


def get_or_create_module(session: Session, repo_id: int, cluster_id: int, module_desc: str) -> int:
    statement = select(Module).where(
        Module.repo == repo_id,
        Module.cluster_id == cluster_id,
        Module.module_desc == module_desc
    )
    result = session.exec(statement).first()
    if result:
        return result.id

    module = Module(repo=repo_id, cluster_id=cluster_id, module_desc=module_desc)
    session.add(module)
    session.commit()
    session.refresh(module)
    return module.id


def get_or_create_feature(session: Session, module_id: int, feature_id_val: int, feature_desc: str) -> int:
    statement = select(Feature).where(
        Feature.module == module_id,
        Feature.feature_id == feature_id_val
    )
    result = session.exec(statement).first()
    if result:
        return result.id

    feature = Feature(module=module_id, feature_id=feature_id_val, feature_desc=feature_desc)
    session.add(feature)
    session.commit()
    session.refresh(feature)
    return feature.id


def get_or_create_code_map(session: Session, feature_id: int, method_name: str) -> int:
    statement = select(CodeMap).where(
        CodeMap.feature == feature_id,
        CodeMap.method_name == method_name
    )
    result = session.exec(statement).first()
    if result:
        return result.id

    code_map = CodeMap(feature=feature_id, method_name=method_name)
    session.add(code_map)
    session.commit()
    session.refresh(code_map)
    return code_map.id


def insert_row(session: Session, row: dict, repo_id: int):
    # row keys: 'cluster_id', 'module_desc', 'id', 'desc', 'method_name'
    # Ensure types are correct
    cluster_id = int(row['cluster_id']) if row['cluster_id'] else 0
    feature_id_val = int(row['id']) if row['id'] else 0

    module_id = get_or_create_module(session, repo_id, cluster_id, row['module_desc'])
    feature_id = get_or_create_feature(session, module_id, feature_id_val, row['desc'])
    get_or_create_code_map(session, feature_id, row['method_name'])


def save_features(session: Session, project_id: int, output_dir: str):
    file_path = os.path.join(output_dir, "features.csv")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist.")
        return

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            insert_row(session, row, project_id)


def get_or_create_graph_edge(session: Session, src: str, dest: str, repo_id: int) -> int:
    statement = select(GraphEdge).where(
        GraphEdge.src == src,
        GraphEdge.dest == dest,
        GraphEdge.repo == repo_id
    )
    result = session.exec(statement).first()
    if result:
        return result.id

    edge = GraphEdge(src=src, dest=dest, repo=repo_id)
    session.add(edge)
    session.commit()
    session.refresh(edge)
    return edge.id


def save_edges(session: Session, project_id: int, output_dir: str):
    file_adj_path = os.path.join(output_dir, "file_adj_matrix.csv")
    if not os.path.exists(file_adj_path):
        print(f"Warning: {file_adj_path} does not exist.")
        return

    file_path = os.path.join(output_dir, "files.csv")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist.")
        return

    # Load adjacency matrix CSV
    # Assuming first row and first column are node names
    try:
        df_adj = pd.read_csv(file_adj_path, header=None)
        df_files = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_adj_path}: {e}")
        return

    adj_values = df_adj.values
    name_map = df_files.set_index('id')['longname'].to_dict()

    coords = np.argwhere(adj_values == 1)
    for x, y in coords:
        src = name_map.get(x, "Unknown")
        dst = name_map.get(y, "Unknown")
        get_or_create_graph_edge(session, src, dst, project_id)


def set_finish(session: Session, project_id: int):
    project = session.get(ProjectInfo, project_id)
    if project:
        project.summary_flag = True
        session.add(project)
        session.commit()


def save_summary_to_db(project_id: int, output_dir: str):
    """
    Main entry point to save summary data to database.
    """
    with Session(engine) as session:
        print(f"Starting save_summary_to_db for project {project_id} from {output_dir}")
        clear_project_data(session, project_id)
        save_features(session, project_id, output_dir)
        save_edges(session, project_id, output_dir)
        set_finish(session, project_id)
        print(f"Finished save_summary_to_db for project {project_id}")


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    project_id = 70
    output_dir = os.path.join(here, "out", str(project_id))

    save_summary_to_db(project_id, output_dir)
