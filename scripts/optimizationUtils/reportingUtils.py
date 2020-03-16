import pandas

def displayReport(resList):
    print('Solution report')
    print(resList[0].to_string())
    print('---------------------------------------------------')


def saveReport(resList, reportFileName, bidFileName):
    with pandas.ExcelWriter(reportFileName) as writer:
        resList[0].to_excel(writer, sheet_name='Main_results')
        resList[1].to_excel(writer, sheet_name='Bid')
        resList[2].to_excel(writer, sheet_name='Profit-distribution')
        resList[3].to_excel(writer, sheet_name='Imbalance_volumes')
        resList[4].to_excel(writer, sheet_name='Expected_hourly_imbalance')

    resList[1].to_csv(bidFileName, header=False)