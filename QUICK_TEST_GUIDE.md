# 🧪 QUICK TEST GUIDE - Copy & Paste Ready

## 🚀 Your app is running at: http://localhost:8501

---

## ✅ **QUICK 5-MINUTE TEST** (Essential Posts)

### 1️⃣ **High-Risk Test** (MUST show Depression-Risk)
```
I hate myself so much, everything I do ends in failure.
```
**Expected:** 
- ✅ Prediction: "High Depression-Risk Language"
- ✅ Tokens highlighted: hate, myself, failure
- ✅ Emotional intensity: 0.65-0.85

---

### 2️⃣ **Crisis Test** 🚨 (MUST show RED WARNING)
```
I don't want to live anymore.
```
**Expected:**
- 🚨 **RED CRISIS ALERT BOX**
- 🚨 Hotline: 988
- 🚨 Emotional intensity: 0.85-1.00
- 🚨 LLM detects suicidal ideation

---

### 3️⃣ **Neutral Test** (MUST show Control)
```
Just finished my homework, time to watch a movie.
```
**Expected:**
- ✅ Prediction: "Control" (Non-Depression)
- ✅ Low probability for depression
- ✅ Emotional intensity: 0.00-0.10

---

### 4️⃣ **Ambiguous Test** (MUST show uncertainty warning)
```
I'm fine.
```
**Expected:**
- ⚠️ Ambiguity warning (low confidence)
- ⚠️ "Human review recommended"
- ⚠️ LLM notes "Insufficient context"

---

### 5️⃣ **Sarcasm Test** (LLM should detect hidden negativity)
```
Yeah sure, my life is 'amazing'. Totally loving it.
```
**Expected:**
- ✅ LLM reasoning detects sarcasm
- ✅ Notes "ironic tone" or "hidden negativity"

---

## 📋 **ALL 22 TEST POSTS** (Complete Testing)

### 🚨 Category 1: High Depression-Risk (4 posts)
```
1. I hate myself so much, everything I do ends in failure.
2. Nothing feels worth it anymore. I feel empty every single day.
3. I can't find any reason to keep going like this.
4. Every day feels heavier than the last. I'm tired of pretending.
```

### 🟠 Category 2: Moderate Distress (3 posts)
```
5. I feel exhausted all the time and I can't focus on anything.
6. My motivation is completely gone, I don't enjoy the things I used to.
7. I'm struggling a lot lately, but I'm not sure what's wrong with me.
```

### 🟡 Category 3: Low-Risk Sadness (3 posts)
```
8. I'm feeling a bit down today, but I think I'll be okay.
9. It's been a rough week, just trying to push through everything.
10. I feel lonely sometimes, but I try to distract myself.
```

### ⚪ Category 4: Neutral Control (3 posts)
```
11. Just finished my homework, time to watch a movie.
12. I'm going for a walk later, the weather is nice today.
13. Made pasta today, turned out pretty good!
```

### 🔥 Category 5: Ambiguous Cases (3 posts)
```
14. I'm done with everything… who even cares anymore.
15. I failed my test again. What is even the point?
16. People keep telling me to be strong, but I don't feel strong.
```

### 🎭 Category 6: Sarcastic/Hidden (3 posts)
```
17. Yeah sure, my life is 'amazing'. Totally loving it.
18. Guess I'm just the family disappointment, lol.
19. I'm fine.
```

### 🚨 Category 7: CRISIS POSTS ⚠️ (3 posts)
```
20. I don't want to live anymore.
21. I'm thinking of ending everything tonight.
22. No one would miss me if I disappeared.
```

---

## ✅ **WHAT TO CHECK FOR EACH POST:**

### Step 1: Classification
- [ ] Shows prediction (Depression-Risk or Control)
- [ ] Shows confidence percentage
- [ ] Shows class probabilities
- [ ] Risk level indicator (High/Moderate/Low)

### Step 2: Token Explanation
- [ ] Top 10 important words highlighted
- [ ] RED background color visible
- [ ] Full words (not character fragments)
- [ ] Words make sense (hate, myself, failure)

### Step 3: Text Preprocessing
- [ ] Shows original vs cleaned text
- [ ] Reports preprocessing changes

### Step 4: LLM Reasoning
- [ ] Emotional Intensity Analysis (0.00-1.00)
- [ ] Confidence level (High/Moderate/Low)
- [ ] Risk assessment
- [ ] Key phrases identified
- [ ] Emotional signals detected
- [ ] Clinical symptoms reflected
- [ ] Cognitive patterns noted
- [ ] DSM-5 clinical context
- [ ] Critical disclaimer

### Step 5: Ambiguity Check
- [ ] Shows confidence level
- [ ] Warnings for <60% confidence
- [ ] Human review recommendation

### Step 6: Final Summary
- [ ] Overall interpretation
- [ ] Suggested action
- [ ] Limitations noted

### Export Feature
- [ ] Download TXT Report button
- [ ] Download CSV Data button
- [ ] Files download successfully

### Crisis Detection (Posts 20-22)
- [ ] 🚨 RED CRISIS ALERT appears
- [ ] Hotline numbers displayed (988, 741741)
- [ ] International resources shown
- [ ] Emergency warning prominent

### Safety & Ethics
- [ ] "Depression-Risk Language" (not "Detected")
- [ ] Footer disclaimer visible
- [ ] Crisis resources at bottom
- [ ] No diagnostic claims

---

## 🐛 **KNOWN FIXES APPLIED:**

✅ **Crisis Detection Enhanced:**
- Added phrases: "don't want to live", "live anymore", "ending everything", "disappeared", "miss me if"
- Now catches all 3 crisis test posts

✅ **Accessibility Fix:**
- Fixed empty label warnings in text_area components
- Added label_visibility="hidden"

✅ **Model Loading:**
- Models located in: `models/trained/`
- All 5 models should load successfully

---

## 🎯 **PASS CRITERIA:**

### CRITICAL (Must Pass):
- [ ] Crisis posts show RED WARNING
- [ ] High-risk posts get Depression-Risk prediction
- [ ] Neutral posts get Control prediction
- [ ] Token highlighting shows real words
- [ ] Export buttons work
- [ ] BERT-Base shows 88% accuracy (not 0.0%)

### IMPORTANT (Should Pass):
- [ ] Emotional intensity scoring works
- [ ] LLM reasoning is comprehensive
- [ ] Ambiguity warnings appear
- [ ] All 5 models load
- [ ] Compare tab works

---

## 📊 **TEST RESULTS TEMPLATE:**

```
POST: [paste post here]
CATEGORY: [High-Risk / Moderate / Low-Risk / Neutral / Ambiguous / Sarcastic / Crisis]

✅ PASS / ❌ FAIL

Classification:
- Prediction: _______
- Confidence: ______%
- Expected: _______

Token Highlighting:
- Top words: _______
- Visual: ✅ / ❌

LLM Reasoning:
- Emotional intensity: _______
- Symptoms detected: _______
- Quality: ✅ / ❌

Crisis Detection (if applicable):
- Red warning: ✅ / ❌
- Hotlines shown: ✅ / ❌

Export:
- TXT download: ✅ / ❌
- CSV download: ✅ / ❌

NOTES: _______
```

---

## 🚀 **START TESTING NOW:**

1. **Open:** http://localhost:8501
2. **Copy** post #1 (hate myself)
3. **Paste** into "Enter text to analyze"
4. **Click** "Analyze Text"
5. **Verify** all features work
6. **Repeat** for all 22 posts

---

## 📝 **REPORT BUGS:**

If you find issues:
1. Note which post caused the issue
2. Screenshot the error
3. Check console/terminal output
4. Document expected vs actual behavior

---

**Last Updated:** November 25, 2025, 8:45 PM  
**App Status:** ✅ Running at http://localhost:8501  
**Fixes Applied:** Crisis detection enhanced, accessibility warnings fixed

**Happy Testing!** 🎉
