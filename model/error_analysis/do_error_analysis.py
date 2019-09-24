model = get_model_info(model_path)
metadata.model_name = model.name
metadata.trained_on = model.train_data
metadata.epochs = model.epochs
metadata.arc_accuracy = model.arc_accuracy


metadata.eval_data = "path/to/eval/data/parsed/with/this/model"
metadata.gold_data = "path/to/gold/data"
evaluator = evaluate.Evaluator(metadata.gold_data, metadata.eval_data)
model_eval = ModelEval()
model_eval = evaluator.Evaluate("all")

