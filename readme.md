# Тестовое задание. Проект Code representation learning

В качестве тестового задания вам предлагается поучаствовать в маленьком ML-contest и решить задачку из теории вероятности:

1) обучите классификатор, который будет предсказывать библиотеку по заголовку вопроса со StackOverflow.

   Датасет для обучения: `train.csv`. Датасет содержит колонки: `title` - заголовок вопроса; `lib` - имя библиотеки, которое нужно предсказать.

   Качество вашего классификатора будет оцениваться метрикой Precision с micro average на закрытом тестовом датасете: `test.csv`.

   Как оформить:
   1) предсказание лучшей вашей модели на закрытом тестовом датасете должно лежать в файле: `submission.csv`. Cледующего формата:
     
      | id  | lib |
      |-----|--------|
      | 84026| numpy  |
      | 8930| django |
      | ... | ...    |

   2) скрипт `train_script.py` для воспроизведения результата, который обучает модель и кладёт в текущую директорию файл `./submission.csv`.
     Все необходимые библиотеки для его работы укажите в `requirements.txt`. Не забудьте для получения похожих результатов зафиксировать random seeds.
     Ожидается, что по коду можно будет полностью отследить процесс решения задачи (например, подготовка данных, извлечение и отбор фитчей, подбор гиперпараметров и т.д.).

   Ограничение: используйте пожалуйста `python 3.9` и именование файлов, как указано в задании, для облегчения проверки. Можно использовать pre-trained models


2) Прибор для выявления брака на фабрике имеет вероятность ошибки 5% (и первого и второго рода), процент брака составляет 5% от всего объёма выпускаемой продукции.

   1) Какая вероятность того, что мы выявили брак, если прибор выдал положительный результат - "продукция бракованная"?

   2) Почему же в жизни все-таки используют такие приборы? Что можно было бы изменить в процедуре поиска брака, не меняя точности прибора, так, чтобы вероятность из первого вопроса P(брак|"+") выросла?

   3) Какое соотношение можно вывести между процентом брака P(брак) и ошибкой прибора, если мы хотим , чтобы прибор работал лучше честной монетки, хуже или также?

# Решение
Описание идеи:

Извлекаю фичи с помощью BERT, затем конкатенирую 4 последних скрытых слоя для токена [CLS] и дополнительные фичи. Дополнительные фичи - присутствие в тексте названий библиотек или их синонимов (pandas и pd - одна и та же фича). Затем передаю это в классификатор [768*4 + 24 -> 256 -> 64 -> 24] с BatchNorm и Dropout между слоями. На валидационной выборке получил precision 0.6331.
