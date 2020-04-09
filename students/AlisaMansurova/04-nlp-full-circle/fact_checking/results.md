### Домен

Всі альбоми з роками випуску одного із ~~найвеличніших лідерів сучасності~~ найвпливовіших на метал-сцену гуртів - Black Sabbath

Хотілося до років випуску додати ще лейбл та учасників (і власне спочатку я їх додала, але вже ресурсів на парсинг не вистачило, тому поки це в списку на "доробити")

[кверя](./sparql_query), [результати квері](./dbpedia_query_results.json)

### Метрики

Рахувала кількість співпадінь, фолс-позітівів і фолс-негатівів. Також у деяких альбомів є нюанс - перевипуск у різні роки. Для своєї метрики я вважала, що співпадіння назви альбому достатньо і додавала `weight` для кожного альбому залежно від кількості співпадінь по рокам (якщо співпала і назва, і роки - `1`, якщо з років співпало `N`, то ставимо `weight = N/len(expected_years)`)

### Результати

Результати виявились досить невтішними в основному через складність пошуку назв альбомів, так як `spacy`-вський NER у переважній більшості не працював (пробувала `en_core_web_lg`, але не допомогло), і довелось обробляти текст за допомогою костилів і такої-то матері. Що вийшло, можна подивитись [тут](./results.json)