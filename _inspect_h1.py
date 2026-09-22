from app.database import SessionLocal
from app.models import HypothesisRevision
db=SessionLocal()
try:
    r=db.query(HypothesisRevision).filter(HypothesisRevision.hypothesis_id==1, HypothesisRevision.revision_number==2).first()
    print("WHEN", r.when_data if r else None)
    print("DO", r.do_data if r else None)
    print("EXPECT", r.expect_data if r else None)
finally:
    db.close()
