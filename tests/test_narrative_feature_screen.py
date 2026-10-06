import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analyses'))
from narrative_feature_screen import FEATURES, validate
from narrative_dialogue_batch import budget_allows
from narrative_broad_audit import followup_errors


class ScreenValidationTests(unittest.TestCase):
    def setUp(self):
        self.source=[{'id':'a','text':'An exact quotation.'}]
        self.obj={'myths':[{'id':'a','readable':True,'absent':list(FEATURES),'evidence':[]}]}

    def test_all_absent_has_explicit_coverage(self):
        self.assertEqual(validate(self.obj,self.source),[])

    def test_missing_category_is_not_absent(self):
        self.obj['myths'][0]['absent'].pop()
        self.assertTrue(validate(self.obj,self.source))

    def test_duplicate_id_rejected(self):
        self.obj['myths'].append(copy.deepcopy(self.obj['myths'][0]))
        self.assertTrue(validate(self.obj,self.source))

    def test_unreadable_not_negative(self):
        self.obj['myths'][0]['readable']=False
        self.assertTrue(validate(self.obj,self.source))
        self.obj['myths'][0]['absent']=[]
        self.assertEqual(validate(self.obj,self.source),[])

    def evidence(self):
        row=self.obj['myths'][0];row['absent'].remove('challenge_dialogue')
        row['evidence']=[{'yes':['challenge_dialogue'],'unclear':[],
                         'form':'event','stance':'unclear','q':'An exact quotation.'}]
        return row['evidence'][0]

    def test_exact_quote_passes(self):
        self.evidence()
        self.assertEqual(validate(self.obj,self.source),[])

    def test_normalized_quote_rejected(self):
        self.evidence()['q']='An  exact quotation.'
        self.assertTrue(validate(self.obj,self.source))

    def test_absent_evidence_conflict(self):
        self.evidence();self.obj['myths'][0]['absent'].append('challenge_dialogue')
        self.assertTrue(validate(self.obj,self.source))

    def test_joined_form_rejected(self):
        self.evidence()['form']='hypothetical/counterfactual'
        self.assertTrue(validate(self.obj,self.source))

    def test_unclear_is_separate(self):
        e=self.evidence();e['unclear']=e.pop('yes');e['yes']=[]
        self.assertEqual(validate(self.obj,self.source),[])

    def test_pending_reservations_count(self):
        self.assertFalse(budget_allows(1,[{'bound':60}],40))
        self.assertTrue(budget_allows(1,[{'bound':60}],39))

    def test_settled_usage_releases_only_unused_reservation(self):
        self.assertTrue(budget_allows(1,[{'bound':60,'accounted_upper_usd':10}],89))
        self.assertFalse(budget_allows(1,[{'bound':60,'accounted_upper_usd':10}],90))

    def test_lower_cap_is_enforced(self):
        self.assertFalse(budget_allows(0,[{'bound':25}],2,26))
        self.assertTrue(budget_allows(0,[{'bound':25}],1,26))

    def test_followup_changed_rule_needs_two_rounds(self):
        t={'trajectory_id':'A001','myths':[{'round':1,'text':'Always give.'},{'round':2,'text':'Give if they return.'}]}
        obj={'trajectory_id':'A001','reservations':'AI interpretation', 'findings':[
            {'certainty':'clear','change_type':'changed_prescription','evidence':[{'round':2,'quote':'Give if they return.'}]}]}
        self.assertIn('change needs earlier and later evidence',followup_errors(obj,t))
        obj['findings'][0]['evidence'].append({'round':1,'quote':'Always give.'})
        self.assertEqual(followup_errors(obj,t),[])

    def test_followup_invented_quote_rejected(self):
        t={'trajectory_id':'A001','myths':[{'round':1,'text':'Always give.'}]}
        obj={'trajectory_id':'A001','reservations':'', 'findings':[
            {'certainty':'unclear','change_type':'baseline','evidence':[{'round':1,'quote':'Punish defectors.'}]}]}
        self.assertIn('quote',followup_errors(obj,t))


if __name__=='__main__':unittest.main()
