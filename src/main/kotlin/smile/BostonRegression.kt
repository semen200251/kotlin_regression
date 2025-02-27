package smile

import smile.regression.RandomForest
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import org.apache.commons.csv.CSVFormat


fun main() {
    // Читаем датасет о ценах на жилье
    val housing = Read.csv("src/main/resources/BostonHousing.csv", CSVFormat.DEFAULT.withFirstRecordAsHeader())

    println(housing)

    // Формируем зависимую переменную (что предсказываем)
    val formula = Formula.lhs("medv")


    // Кросс-валидация для оценки качества модели
    val res = CrossValidation.regression(
        10, formula, housing,
        { f, data -> RandomForest.fit(f, data) })

    println(res)
}