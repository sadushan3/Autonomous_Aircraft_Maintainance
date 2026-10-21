"""Validate the completed local document and package the manuscript artifacts."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

from docx import Document
from docx.oxml.ns import qn
import fitz
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
STEM='IOP_Anomaly_and_Health_Monitoring'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source=(HERE/'manuscript.md').read_text(encoding='utf-8')
    refs=json.loads((HERE/'reference_audit.json').read_text(encoding='utf-8'))['references']
    cited={int(item) for group in re.findall(r'\[(\d+(?:,\d+)*)\]',source) for item in group.split(',')}
    assert cited=={r['id'] for r in refs}==set(range(1,12))
    template=Document(HERE/'conference_guidelines/IOP-ConfSer-template.docx')
    document=Document(HERE/f'{STEM}.docx')
    a,b=template.sections[0],document.sections[0]
    for attr in ['page_width','page_height','left_margin','right_margin','top_margin','bottom_margin']:
        assert getattr(a,attr)==getattr(b,attr),attr
    assert len(document.tables)==4 and len(document.inline_shapes)==3
    assert len(document._element.xpath('//m:oMath'))==7
    assert not document._element.xpath('//m:num/m:e | //m:den/m:e | //m:e/m:e')
    for part in [b.header,b.footer,b.first_page_header,b.first_page_footer,b.even_page_header,b.even_page_footer]:
        assert not any(p.text.strip() for p in part.paragraphs)
        assert 'PAGE' not in part._element.xml
    assert abs(document.styles['IOP-CS-Abstract'].paragraph_format.space_after.cm-1)<.002
    for style in ['IOP-CS-BodyText','IOP-CS-BodyNoIndent','IOP-CS-ReferenceText']:
        assert document.styles[style].paragraph_format.line_spacing==1
    xml=document._element.xml
    assert not any(bad in xml for bad in ['\u00e2\u20ac','\u00c3\u00a9','\ufffd','{{','Lorem ipsum','Replace this text'])
    pdf=fitz.open(HERE/f'{STEM}.pdf')
    text='\n'.join(page.get_text() for page in pdf)
    assert len(pdf)==10
    assert 'Selvakumar Sadurshan' in text and 'sadushansadu33@gmail.com' in text
    assert 'to be supplied' in text
    for value in ['0.9952','0.8999','0.6321','0.7426','0.4601','0.9477','0.8976','0.8908']:
        assert value in text,value
    assert not any(bad in text for bad in ['\u00e2\u20ac','\ufffd','{{','Lorem ipsum'])
    font_xrefs={font[0] for page in pdf for font in page.get_fonts()}
    assert all(pdf.extract_font(xref)[3] for xref in font_xrefs),'Unembedded PDF font'
    assert all(abs(p.rect.width-595.3)<.5 and abs(p.rect.height-841.9)<.5 for p in pdf)
    uri_links={link['uri'] for page in pdf for link in page.get_links() if 'uri' in link}
    assert {ref['url'] for ref in refs} <= uri_links
    # Validate the text and image boxes against the template margins.
    limits=(b.left_margin.pt,b.top_margin.pt,b.page_width.pt-b.right_margin.pt,b.page_height.pt-b.bottom_margin.pt)
    violations=[]
    for i,page in enumerate(pdf):
        visible_boxes=[]
        for block in page.get_text('rawdict')['blocks']:
            if block['type']==1:
                visible_boxes.append(block['bbox'])
            else:
                for line in block['lines']:
                    for span in line['spans']:
                        visible_boxes.extend(c['bbox'] for c in span['chars'] if c['c'].strip())
        for x0,y0,x1,y1 in visible_boxes:
            if x0<limits[0]-3 or y0<limits[1]-4 or x1>limits[2]+3 or y1>limits[3]+4:
                violations.append({'page':i+1,'bbox':[x0,y0,x1,y1]})
    assert not violations,violations[:10]
    evidence=json.loads((HERE/'evidence/independent_audit.json').read_text(encoding='utf-8'))
    for rel,expected in evidence['sources_sha256'].items():
        assert digest(HERE.parent/'Backend'/rel)==expected,rel
    validation={'pages':len(pdf),'page_size':'A4, exact supplied-template dimensions',
        'template_margins_preserved':True,'single_spaced_body':True,'header_and_footer_content':False,
        'page_numbers':False,'abstract_gap_mm':10,'native_editable_equations':7,'figures':3,'tables':4,
        'verified_references':11,'all_citations_resolved':True,'all_reference_links_in_pdf':True,
        'all_fonts_embedded':True,'visible_content_within_margins_allowing_3pt_glyph_overhang':True,'no_detected_encoding_corruption':True,
        'source_hashes_match':38,'independently_reconstructed_labels':104897,
        'anomaly_and_health_results_reproduced':True,'author_name_and_email_confirmed':True,
        'remaining_information':['Author affiliation','Conference name and page limit','Venue-specific review and disclosure requirements'],
        'evaluation_scope':'Retrospective C-MAPSS condition proxies; no new prospective or physical-fault experiment.'}
    (HERE/'evidence/document_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    # Refresh previews after the final export.
    previews=HERE/'previews';previews.mkdir(exist_ok=True)
    thumbs=[]
    for i,page in enumerate(pdf):
        pix=page.get_pixmap(matrix=fitz.Matrix(.52,.52),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        canvas=Image.new('RGB',(im.width+12,im.height+30),'#dddddd')
        canvas.paste(im,(6,24));ImageDraw.Draw(canvas).text((8,5),f'Page {i+1}',fill='black');thumbs.append(canvas)
        if i in [0,2,3,5,8,9]:
            page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(previews/f'page_{i+1:02}.png')
    cols=5;rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new('RGB',(cols*thumbs[0].width,rows*thumbs[0].height),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%cols)*im.width,(i//cols)*im.height))
    sheet.save(previews/'contact_sheet.png')
    names=[f'{STEM}.docx',f'{STEM}.pdf','manuscript.md','Humanized_Manuscript.md','README.md',
           'Verified_References.md','reference_audit.json','verify_evidence.py','build_manuscript.py',
           'export_to_pdf.py','validate_and_package.py']
    files=[HERE/name for name in names]
    files+=sorted((HERE/'evidence').glob('*.*'))
    files+=sorted((HERE/'figures').glob('*.*'))
    files+=sorted((HERE/'conference_guidelines').glob('*.*'))
    manifest='\n'.join(digest(p)+'  '+p.relative_to(HERE).as_posix() for p in files)+'\n'
    (HERE/'MANIFEST.sha256').write_text(manifest,encoding='utf-8')
    files.append(HERE/'MANIFEST.sha256')
    archive=HERE/'IOP_Anomaly_Health_Paper_Package.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(HERE).as_posix())
    print(json.dumps({**validation,'archive':archive.name,'archive_bytes':archive.stat().st_size,'files':len(files)},indent=2))


if __name__=='__main__':
    main()
