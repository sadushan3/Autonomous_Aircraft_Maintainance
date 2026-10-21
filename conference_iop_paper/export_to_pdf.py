"""Export the supplied-template DOCX through Microsoft Word in a hidden instance."""
from pathlib import Path
import json
import pythoncom
import win32com.client

here=Path(__file__).resolve().parent
source=here/'IOP_Anomaly_and_Health_Monitoring.docx'
target=source.with_suffix('.pdf')
pythoncom.CoInitialize()
word=None
document=None
try:
    word=win32com.client.DispatchEx('Word.Application')
    word.Visible=False
    word.DisplayAlerts=0
    word.AutomationSecurity=3
    document=word.Documents.Open(str(source),ConfirmConversions=False,ReadOnly=True,AddToRecentFiles=False,Visible=False)
    document.Repaginate()
    pages=document.ComputeStatistics(2)
    document.ExportAsFixedFormat(OutputFileName=str(target),ExportFormat=17,OpenAfterExport=False,
        OptimizeFor=0,Range=0,Item=0,IncludeDocProps=True,CreateBookmarks=0,DocStructureTags=True,
        BitmapMissingFonts=False,UseISO19005_1=True)
    print(json.dumps({'pdf':str(target),'pages':pages,'renderer':'Microsoft Word','pdf_a_requested':True}))
finally:
    if document is not None:
        document.Close(SaveChanges=0)
    if word is not None:
        word.Quit(SaveChanges=0)
    pythoncom.CoUninitialize()
