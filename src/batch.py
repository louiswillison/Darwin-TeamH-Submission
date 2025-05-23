import os
import sys
import time
from datetime import datetime

from tqdm import tqdm

import feature_extract

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Batch:
    def __init__(self, args):
        self.args = args
        self.queue = []


class Run:
    def __init__(self, audio_type, model, features):
        self.audio_type = audio_type
        self.model = model
        self.features = features
        self.status = "Queued"
        self.start_time = None
        self.end_time = None
        self.roc_output = None
        self.cm_output = None
        self.result = None


class UILogic:
    def __init__(self):
        self.main_choice = None
        self.audio_type = None
        self.model = None
        self.features = None
        self.sort_by = None
        self.sort_order = None
        self.history = []  # ui history for back navigation


batch = Batch(args=None)
ui_logic = UILogic()

AUDIO_TYPES = {
    "1": "Cough",
    "2": "Speech",
    "3": "Combined",
}

MODELS = {
    "1": "Random Forest",
    "2": "Extra Trees",
    "3": "Gradient Boosting",
    "4": "Logistic Regression",
    "5": "SVM",
    "6": "CNN",
    "7": "CNN-LSTM",
    "8": "FFNN",
    "9": "ResNet50",
}

MODELS_INTERNAL = {
    "1": "rf",
    "2": "et",
    "3": "gb",
    "4": "lr",
    "5": "svm",
    "6": "cnn",
    "7": "lstm",
    "8": "ffnn",
    "9": "resnet",
}

FEATURE_SETS = {
    "1": {  # cough
        "1": "MFCC",
        "2": "GeMAPS",
        "3": "ComParE",
    },
    "2": {  # speech
        "1": "GeMAPS",
        "2": "ComParE",
    },
    "3": {  # both
        "1": "GeMAPS Cough + GeMAPS Speech",
        "2": "ComParE Cough + ComParE Speech",
        "3": "ComParE Cough + GeMAPS Speech",
        "4": "GeMAPS Cough + ComParE Speech",
    },
}


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def pre_checks():
    features_files = [
        "features_cough_mfcc.csv",
        "features_cough_gemaps.csv",
        "features_cough_compare.csv",
        "features_speech_gemaps.csv",
        "features_speech_compare.csv",
        "features_cough_gemaps_speech_gemaps.csv",
        "features_cough_compare_speech_compare.csv",
        "features_cough_compare_speech_gemaps.csv",
        "features_cough_gemaps_speech_compare.csv",
    ]
    features = [
        "COUGH_MFCC",
        "COUGH_GeMAPS",
        "COUGH_COMPARE",
        "SPEECH_GeMAPS",
        "SPEECH_COMPARE",
        "COUGHGEspeechGE",
        "COUGHCOspeechCO",
        "COUGHCOspeechGE",
        "COUGHGEspeechCO",
    ]
    for file in features_files:
        if not os.path.exists("features/"):
            os.makedirs("features/")
        if not os.path.exists("features/" + file):
            print(f"Missing {file}, generating...")
            feature_extract.FeatureExtractor(
                use_subset=False,
                output_path="features/" + file,
                batch_size=1000,
                verbose=False,
                feature_type=features[features_files.index(file)],
            ).extract_audio_files(web=False, ios=True, android=True)
    clear_terminal()


def ui_main_menu():
    clear_terminal()
    print("**************************************")
    print("Welcome to the Team H Batch Processing UI")
    print("Current runs in queue: " + str(len(batch.queue)))
    print("**************************************")
    print("1) Add run to queue")
    print("2) Remove run from queue")
    print("3) View queue")
    print("4) Process queue")
    print("5) Import queue from file")
    print("6) Export queue to file")
    print("7) View top results")
    print("0) Exit")
    print("")
    choice = input("Select an option: ")
    if choice == "1":
        ui_logic.history.append(ui_main_menu)
        ui_audio_type()
    elif choice == "2":
        ui_logic.history.append(ui_main_menu)
        ui_remove_from_queue()
    elif choice == "3":
        ui_logic.history.append(ui_main_menu)
        ui_view_queue()
    elif choice == "4":
        ui_logic.history.append(ui_main_menu)
        ui_process_queue()
    elif choice == "5":
        ui_logic.history.append(ui_main_menu)
        ui_import_queue()
    elif choice == "6":
        ui_logic.history.append(ui_main_menu)
        ui_export_queue()
    elif choice == "7":
        ui_logic.history.append(ui_main_menu)
        ui_view_top_results_sort_by()
    elif choice == "0":
        sys.exit()
    else:
        input("Option not implemented yet. Press Enter to return...")
        ui_main_menu()


def ui_audio_type():
    clear_terminal()
    print("**************************************")
    print("Select audio type")
    print("1) Cough")
    print("2) Speech")
    print("3) Combined")
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    elif choice in ["1", "2", "3"]:
        ui_logic.audio_type = choice
        ui_logic.history.append(ui_audio_type)
        ui_model_selection()
    else:
        ui_audio_type()


def ui_model_selection():
    clear_terminal()
    print("**************************************")
    print("Select model")
    print("1) Random Forest")
    print("2) Extra Trees")
    print("3) Gradient Boosting")
    print("4) LR")
    print("5) SVM")
    print("6) CNN")
    print("7) CNN-LSTM")
    print("8) FFNN")
    print("9) ResNet50")
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    elif choice in [str(i) for i in range(1, 10)]:
        ui_logic.model = choice
        ui_logic.history.append(ui_model_selection)
        ui_feature_selection()
    else:
        ui_model_selection()


def ui_feature_selection():
    clear_terminal()
    print("**************************************")
    print("Select features")
    if ui_logic.audio_type == "1":
        print("1) MFCC")
        print("2) GeMAPS")
        print("3) Compare")
    elif ui_logic.audio_type == "2":
        print("1) GeMAPS")
        print("2) Compare")
    else:
        print("1) GeMAPS Cough and GeMAPS Speech")
        print("2) Compare Cough and Compare Speech")
        print("3) Compare Cough and GeMAPS Speech")
        print("4) GeMAPS Cough and Compare Speech")
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    else:
        ui_logic.features = choice
        run = Run(ui_logic.audio_type, ui_logic.model, ui_logic.features)
        batch.queue.append(run)
        print("\nRun added to queue successfully.")
        input("Press Enter to return to main menu...")
        ui_logic.history.clear()
        ui_main_menu()


def ui_view_queue():
    clear_terminal()
    print("**************************************")
    print("Current runs in queue:")
    if not batch.queue:
        print("No runs in queue.")
    else:
        for i, run in enumerate(batch.queue):
            if run.audio_type == "1":
                print(
                    f"  {i + 1}) {AUDIO_TYPES.get(run.audio_type)} - {MODELS.get(run.model)} - {FEATURE_SETS['1'].get(run.features)}"
                )
            elif run.audio_type == "2":
                print(
                    f"  {i + 1}) {AUDIO_TYPES.get(run.audio_type)} - {MODELS.get(run.model)} - {FEATURE_SETS['2'].get(run.features)}"
                )
            else:
                print(
                    f"  {i + 1}) {AUDIO_TYPES.get(run.audio_type)} - {MODELS.get(run.model)} - {FEATURE_SETS['3'].get(run.features)}"
                )
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    else:
        ui_view_queue()


def ui_remove_from_queue():
    clear_terminal()
    print("**************************************")
    print("Remove run from queue:")
    if not batch.queue:
        print("No runs in queue.")
    else:
        for i, run in enumerate(batch.queue):
            if run.audio_type == "1":
                print(
                    f"{i + 1}) {AUDIO_TYPES.get(run.audio_type)} - {MODELS.get(run.model)} - {FEATURE_SETS['1'].get(run.features)}"
                )
            elif run.audio_type == "2":
                print(
                    f"{i + 1}) {AUDIO_TYPES.get(run.audio_type)} - {MODELS.get(run.model)} - {FEATURE_SETS['2'].get(run.features)}"
                )
            else:
                print(
                    f"{i + 1}) {AUDIO_TYPES.get(run.audio_type)} - {MODELS.get(run.model)} - {FEATURE_SETS['3'].get(run.features)}"
                )
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    else:
        try:
            index = int(choice) - 1
            if 0 <= index < len(batch.queue):
                batch.queue.pop(index)
                print("\nRun removed from queue successfully.")
            else:
                print("\nInvalid choice.")
        except ValueError:
            print("\nInvalid choice.")
        input("Press Enter to return to main menu...")
        ui_logic.history.clear()
        ui_main_menu()


def ui_process_queue():
    clear_terminal()
    print("**************************************")
    print("Processing queue:")

    if not batch.queue:
        print("No runs in queue.")
    else:
        print("\nQueued Runs:")
        print("------------------------")
        for i, run in enumerate(batch.queue):
            audio_label = AUDIO_TYPES.get(run.audio_type, "Unknown")
            model_label = MODELS.get(run.model, "Unknown")
            feature_label = FEATURE_SETS.get(run.audio_type, {}).get(
                run.features, "Unknown"
            )
            status = run.status or "Queued"

            print(
                f"{i + 1}) [{status}] {audio_label} - {model_label} - {feature_label}"
            )

        print("\nStart processing the runs?")
        print("1) Yes")
        print("0) Back")

        choice = input("Select an option: ")
        if choice == "1":
            for i, run in enumerate(batch.queue):
                time.sleep(
                    5
                )  # stops errors if models are too fast (turns out it doesn't but they happen less)
                audio_label = AUDIO_TYPES.get(run.audio_type, "Unknown")
                model_label = MODELS.get(run.model, "Unknown")
                feature_label = FEATURE_SETS.get(run.audio_type, {}).get(
                    run.features, "Unknown"
                )

                print(
                    f"\n>>> Running {i+1}/{len(batch.queue)}: {audio_label} - {model_label} - {feature_label}"
                )
                execute_run(run)
                if run.status == "Failed":
                    print(f"❌ Run failed [took {run.end_time - run.start_time}]")
                    continue
                auc, f1, sens, spec = run.result
                print(
                    f"✅ Done: AUC={auc:.4f}, F1={f1:.4f}, Sens={sens:.4f}, Spec={spec:.4f} [took {run.end_time - run.start_time}]"
                )
            input("\nAll runs complete. Press Enter to return...")
            ui_main_menu()
        elif choice == "0":
            ui_logic.history.pop()()
        else:
            ui_process_queue()


def get_queue_files():
    if not os.path.exists("queues"):
        os.makedirs("queues")
    queue_files = [
        f for f in os.listdir("queues") if os.path.isfile(os.path.join("queues", f))
    ]
    return queue_files


def count_queue_file_runs(filename):
    with open(f"queues/{filename}", "r") as f:
        lines = f.readlines()
    return len(lines)


def import_queue(filename):
    try:
        with open(f"queues/{filename}", "r") as f:
            lines = f.readlines()
        for line in lines:
            audio_type, model, features = line.strip().split(" - ")
            run = Run(audio_type, model, features)
            batch.queue.append(run)
        print(f"Queue imported from {filename}")
    except Exception as e:
        print(f"Error importing queue: {e}")


def ui_import_queue():
    clear_terminal()
    print("**************************************")
    print("Import queue from file:")
    queue_files = get_queue_files()
    if not queue_files:
        print("No queue files found.")
    else:
        for i, file in enumerate(queue_files):
            print(f"{i + 1}) {file} - {count_queue_file_runs(file)} runs")
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    else:
        try:
            index = int(choice) - 1
            if 0 <= index < len(queue_files):
                filename = queue_files[index]
                import_queue(filename)
                print("\nQueue imported successfully.")
            else:
                print("\nInvalid choice.")
        except ValueError:
            print("\nInvalid choice.")
        input("Press Enter to return to main menu...")
        ui_logic.history.clear()
        ui_main_menu()


def export_queue(filename):
    try:
        if not os.path.exists("queues"):
            os.makedirs("queues")
        with open(f"queues/{filename}.txt", "w") as f:
            for run in batch.queue:
                f.write(f"{run.audio_type} - {run.model} - {run.features}\n")
        print(f"Queue exported to {filename}.txt")
    except Exception as e:
        print(f"Error exporting queue: {e}")


def ui_export_queue():
    clear_terminal()
    print("**************************************")
    print("Export queue to file:")
    if not batch.queue:
        print("No runs in queue, cannot export.")
    else:
        filename = input("Enter filename (without extension): ")
        if not filename:
            print("Invalid filename.")
            input("Press Enter to return to main menu...")
            ui_logic.history.clear()
            ui_main_menu()
        else:
            export_queue(filename)
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    else:
        ui_export_queue()


def ui_view_top_results_sort_by():
    clear_terminal()
    print("**************************************")
    print("Sort by:")
    print("1) AUC")
    print("2) F1")
    print("3) Sensitivity")
    print("4) Specificity")
    print("5) Timestamp")
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    elif choice in ["1", "2", "3", "4", "5"]:
        ui_logic.sort_by = choice
        ui_view_top_results_sort_order()
    else:
        ui_view_top_results_sort_by()


def ui_view_top_results_sort_order():
    clear_terminal()
    print("**************************************")
    print("Sort order:")
    print("1) Ascending")
    print("2) Descending")
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    elif choice in ["1", "2"]:
        ui_logic.sort_order = choice
        ui_view_top_results()
    else:
        ui_view_top_results_sort_order()


def ui_view_top_results():
    clear_terminal()
    print("**************************************")
    print("Top results:")
    if not os.path.exists("results_log.csv"):
        print("No results log found.")
        input("Press Enter to return to main menu...")
        ui_logic.history.clear()
        ui_main_menu()
        return

    with open("results_log.csv", "r") as f:
        lines = f.readlines()

    lines = lines[1:]

    if ui_logic.sort_by == "1":
        index = 4  # auc
    elif ui_logic.sort_by == "2":
        index = 5  # f1
    elif ui_logic.sort_by == "3":
        index = 6  # sens
    elif ui_logic.sort_by == "4":
        index = 7  # spec
    elif ui_logic.sort_by == "5":
        index = 0  # time

    sorted_lines = sorted(
        lines,
        key=lambda x: float(x.split(",")[index]),
        reverse=ui_logic.sort_order == "2",
    )

    for i, line in enumerate(sorted_lines[:20]):
        line = line.strip()
        audio_label = AUDIO_TYPES.get(line.split(",")[1], "Unknown")
        model_label = MODELS.get(line.split(",")[2], "Unknown")
        feature_label = FEATURE_SETS[line.split(",")[1]].get(
            line.split(",")[3], "Unknown"
        )
        auc, f1, sens, spec = line.split(",")[4:8]
        print(
            "  [Timestamp: %s] | %-10s | %-20s | %-30s | AUC: %s | F1: %s | Sens: %s | Spec: %s"
            % (
                line.split(",")[0],
                audio_label,
                model_label,
                feature_label,
                auc,
                f1,
                sens,
                spec,
            )
        )
    print("")
    print(
        "Sorting by: "
        + ["AUC", "F1", "Sensitivity", "Specificity", "Timestamp"][
            int(ui_logic.sort_by) - 1
        ]
    )
    print("Sorting order: " + ["Ascending", "Descending"][int(ui_logic.sort_order) - 1])
    print("")
    print("0) Back")
    print("")
    choice = input("Select an option: ")
    if choice == "0":
        ui_logic.history.pop()()
    else:
        ui_view_top_results()


def execute_run(run):
    run.status = "Running"
    run.start_time = datetime.now()

    timestamp = str(time.time()).split(".")[0]
    base_name = f"{timestamp}_{run.audio_type}_{run.model}_{run.features}"

    try:
        auc, f1, sens, spec = train_and_evaluate(
            run.audio_type,
            run.model,
            run.features,
            f"roc_{base_name}.png",
            f"cm_{base_name}.png",
            timestamp,
        )
        run.status = "Completed"
        run.end_time = datetime.now()
        run.result = (auc, f1, sens, spec)
        run.roc_output = f"roc_{base_name}.png"
        run.cm_output = f"cm_{base_name}.png"

        with open("results_log.csv", "a") as f:
            f.write(
                f"{timestamp},{run.audio_type},{run.model},{run.features},{auc:.4f},{f1:.4f},{sens:.4f},{spec:.4f}\n"
            )
    except Exception as e:
        print(f"Error during model execution: {e}")
        run.status = "Failed"
        run.end_time = datetime.now()
        return


def train_and_evaluate(
    audio_type, model, features, roc_filename, cm_filename, timestamp
):

    # i hate all of this but it works

    if audio_type == "1":
        if features == "1":
            feature_file = "features/features_cough_mfcc.csv"  # cough mfcc
            roc_filename = (
                "roc_"
                + MODELS_INTERNAL.get(model)
                + "_cough_mfcc_"
                + timestamp
                + ".png"
            )
            cm_filename = (
                "cm_" + MODELS_INTERNAL.get(model) + "_cough_mfcc_" + timestamp + ".png"
            )
        elif features == "2":
            feature_file = "features/features_cough_gemaps.csv"  # cough gemaps
            roc_filename = (
                "roc_" + MODELS_INTERNAL.get(model) + "_cough_ge_" + timestamp + ".png"
            )
            cm_filename = (
                "cm_" + MODELS_INTERNAL.get(model) + "_cough_ge_" + timestamp + ".png"
            )
        elif features == "3":
            feature_file = "features/features_cough_compare.csv"  # cough compare
            roc_filename = (
                "roc_" + MODELS_INTERNAL.get(model) + "_cough_co_" + timestamp + ".png"
            )
            cm_filename = (
                "cm_" + MODELS_INTERNAL.get(model) + "_cough_co_" + timestamp + ".png"
            )

    elif audio_type == "2":
        if features == "1":
            feature_file = "features/features_speech_gemaps.csv"  # speech gemaps
            roc_filename = (
                "roc_" + MODELS_INTERNAL.get(model) + "_speech_ge_" + timestamp + ".png"
            )
            cm_filename = (
                "cm_" + MODELS_INTERNAL.get(model) + "_speech_ge_" + timestamp + ".png"
            )

        elif features == "2":
            feature_file = "features/features_speech_compare.csv"  # speech compare
            roc_filename = (
                "roc_" + MODELS_INTERNAL.get(model) + "_speech_co_" + timestamp + ".png"
            )
            cm_filename = (
                "cm_" + MODELS_INTERNAL.get(model) + "_speech_co_" + timestamp + ".png"
            )

    else:
        if features == "1":
            feature_file = "features/features_cough_gemaps_speech_gemaps.csv"  # cough gemaps speech gemaps
            roc_filename = (
                "roc_"
                + MODELS_INTERNAL.get(model)
                + "_cough_ge_speech_ge_"
                + timestamp
                + ".png"
            )
            cm_filename = (
                "cm_"
                + MODELS_INTERNAL.get(model)
                + "_cough_ge_speech_ge_"
                + timestamp
                + ".png"
            )

        elif features == "2":
            feature_file = "features/features_cough_compare_speech_compare.csv"  # cough compare speech compare
            roc_filename = (
                "roc_"
                + MODELS_INTERNAL.get(model)
                + "_cough_co_speech_co_"
                + timestamp
                + ".png"
            )
            cm_filename = (
                "cm_"
                + MODELS_INTERNAL.get(model)
                + "_cough_co_speech_co_"
                + timestamp
                + ".png"
            )
        elif features == "3":
            feature_file = "features/features_cough_compare_speech_gemaps.csv"  # cough compare speech gemaps
            roc_filename = (
                "roc_"
                + MODELS_INTERNAL.get(model)
                + "_cough_co_speech_ge_"
                + timestamp
                + ".png"
            )
            cm_filename = (
                "cm_"
                + MODELS_INTERNAL.get(model)
                + "_cough_co_speech_ge_"
                + timestamp
                + ".png"
            )
        elif features == "4":
            feature_file = "features/features_cough_gemaps_speech_compare.csv"  # cough gemaps speech compare
            roc_filename = (
                "roc_"
                + MODELS_INTERNAL.get(model)
                + "_cough_ge_speech_co_"
                + timestamp
                + ".png"
            )
            cm_filename = (
                "cm_"
                + MODELS_INTERNAL.get(model)
                + "_cough_ge_speech_co_"
                + timestamp
                + ".png"
            )

    if model == "1":
        from legacy import run_model

        auc, f1, sens, spec = run_model(
            "rf",
            feature_file,
            roc_filename,
            cm_filename,
        )
    elif model == "2":
        from legacy import run_model

        auc, f1, sens, spec = run_model(
            "et",
            feature_file,
            roc_filename,
            cm_filename,
        )
    elif model == "3":
        from legacy import run_model

        auc, f1, sens, spec = run_model(
            "gb",
            feature_file,
            roc_filename,
            cm_filename,
        )
    elif model == "4":
        from lr import run_model

        auc, f1, sens, spec = run_model(
            feature_file,
            roc_filename,
            cm_filename,
        )
    elif model == "5":
        from SVM import run_model

        auc, f1, sens, spec = run_model(
            feature_file,
            roc_filename,
            cm_filename,
        )
    elif model == "6":
        from cnn2 import run_model

        auc, f1, sens, spec = run_model(
            feature_file,
            roc_filename,
            cm_filename,
        )
    elif model == "7":
        from cnn_lstm import run_model

        auc, f1, sens, spec = run_model(
            feature_file,
            roc_filename,
            cm_filename,
        )
    elif model == "8":
        from ffnn import run_model

        auc, f1, sens, spec = run_model(
            feature_file,
            roc_filename,
            cm_filename,
        )
    elif model == "9":
        from resnet50 import run_model

        auc, f1, sens, spec = run_model(
            feature_file,
            roc_filename,
            cm_filename,
        )
    else:
        print("Invalid model selected.")
        return 0.0, 0.0, 0.0, 0.0
    return auc, f1, sens, spec


if __name__ == "__main__":
    pre_checks()
    ui_main_menu()
