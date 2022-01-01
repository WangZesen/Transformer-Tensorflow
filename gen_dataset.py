from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import hydra
import random
import os

ONES_DIGIT = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
TEN_DIGIT = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS_DIGIT = ['', 'ten', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

def _validate(x: int):
	assert x >= 0 and x < 1000

def _convert_int_to_str(x: int):
	return " ".join(list(str(x)))

def _convert_int_to_text(n: int):
	_validate(n)
	if n <= 10:
		return ONES_DIGIT[n]
	if n < 20:
		return TEN_DIGIT[n % 10]
	if n < 100:
		ret = TENS_DIGIT[n // 10]
		if n % 10:
			ret += ' ' + ONES_DIGIT[n % 10]
		return ret
	ret = ONES_DIGIT[n // 100] + ' hundred'
	if n % 100:
		n = n % 100
		if n == 0:
			pass
		elif n <= 10:
			ret += ' and ' + ONES_DIGIT[n]
		elif n < 20:
			ret += ' and ' + TEN_DIGIT[n % 10]
		else:
			ret += ' and ' + TENS_DIGIT[n // 10]
			if n % 10:
				ret += ' ' + ONES_DIGIT[n % 10]
	return ret

def random_question():
	total = random.randint(0, 999)
	x = random.randint(0, total)
	y = total - x

	_validate(x)
	_validate(y)
	_validate(total)

	if random.random() < 0.5:
		question = f'{_convert_int_to_str(x)} add {_convert_int_to_str(y)} equals '
		if random.random() < 0.5:
			question += f'{_convert_int_to_str(total)}'
			answer = f'Correct. It"s {_convert_int_to_text(total)}'
		else:
			wrong_ans = random.randint(0, 999)
			while wrong_ans == total:
				wrong_ans = random.randint(0, 999)
			question += f'{_convert_int_to_str(wrong_ans)}'
			answer = f'Wrong. It"s {_convert_int_to_text(total)}'
	else:
		question = f'{_convert_int_to_str(total)} subtract {_convert_int_to_str(x)} equals '
		if random.random() < 0.5:
			question += f'{_convert_int_to_str(y)}'
			answer = f'Correct. It"s {_convert_int_to_text(y)}'
		else:
			wrong_ans = random.randint(0, 999)
			while wrong_ans == y:
				wrong_ans = random.randint(0, 999)
			question += f'{_convert_int_to_str(wrong_ans)}'
			answer = f'Wrong. It"s {_convert_int_to_text(y)}'
	return question, answer

@hydra.main(config_path="cfg", config_name='config')
def app(cfg):
	os.makedirs(os.path.dirname(to_absolute_path(cfg.data.train_data.dir)), exist_ok=True)
	with open(to_absolute_path(cfg.data.train_data.dir), 'w') as f:
		for _ in range(cfg.data.train_data.size):
			question, answer = random_question()
			print(question, answer, sep='\t', file=f)

	with open(to_absolute_path(cfg.data.test_data.dir), 'w') as f:
		for _ in range(5000):
			question, answer = random_question()
			print(question, answer, sep='\t', file=f)

if __name__ == '__main__':
	app()

