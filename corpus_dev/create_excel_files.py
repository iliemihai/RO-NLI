import os, sys, json
import xlsxwriter


def write_excel(start, end, sentence2id, id2sentence, mapping):
    filename = "RO-NLI_{}-{}.xlsx".format(start, end)
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    row_format = cell_format = workbook.add_format()
    row_format.set_bottom(1)
    row_format.set_bottom_color('#aaaaaa')
    row_format.set_bg_color('#ffffff')

    header_row_format = cell_format = workbook.add_format()
    header_row_format.set_bottom(1)
    header_row_format.set_bottom_color('#666666')
    header_row_format.set_bg_color('#CCCCCC')
    header_row_format.set_bold(True)

    alt_row_format = cell_format = workbook.add_format()
    alt_row_format.set_bottom(1)
    alt_row_format.set_bottom_color('#aaaaaa')
    alt_row_format.set_bg_color('#f0faff')

    bold = workbook.add_format({'bold': True, 'bg_color':'#f0f0ff'})

    cell_format = workbook.add_format()
    cell_format.set_right(1)
    cell_format.set_right_color('#dddddd')
    cell_format.set_bottom(1)
    cell_format.set_bottom_color('#aaaaaa')

    alt_cell_format = workbook.add_format()
    alt_cell_format.set_right(1)
    alt_cell_format.set_right_color('#dddddd')
    alt_cell_format.set_bg_color('#f0faff')
    alt_cell_format.set_bottom(1)
    alt_cell_format.set_bottom_color('#aaaaaa')

    worksheet.set_default_row(20)
    worksheet.set_column('B:B', 3)
    worksheet.set_column('C:C', 100)
    worksheet.set_column('D:D', 50)


    worksheet.write(0, 0, "#")
    worksheet.write(0, 2, "Propozitie originala")
    worksheet.write(0, 3, "Traducere")
    worksheet.set_row(0, cell_format=header_row_format)
    #worksheet.set_row(0, bold)

    row = 1
    for index in range(start, end):
        worksheet.write(row, 0, index)
        if index % 2 == 0:
            worksheet.set_row(row, cell_format=alt_row_format)
            worksheet.write(row, 2, id2sentence[index], alt_cell_format)
        else:
            worksheet.set_row(row, cell_format=row_format)
            worksheet.write(row, 2, id2sentence[index], cell_format)

        row += 1

    workbook.close()

with open("sentence2id.json", "r", encoding="utf8") as f:
    sentence2id = json.load(f)
    id2sentence = []
    for sentence in sentence2id:
        id2sentence.append(sentence[0])

with open("mapping.json", "r", encoding="utf8") as f:
    mapping = json.load(f)

print("Unique sents: {}, total sentences: {}".format(len(sentence2id), len(mapping)))

write_excel(0,2000, sentence2id, id2sentence, mapping)