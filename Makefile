tensor_board:
	tensorboard --logdir=run1:tensor_log/$(EXAMPLE) --host localhost --port 6010


clean:
	find tensor_log/** -type d -exec rm -rf '{}' \;
