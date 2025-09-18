# TODO List for Flask Website Build

## Step 1: Update Dependencies
- [x] Add reportlab to requirements.txt for PDF generation

## Step 2: Update Styling
- [x] Integrate Bootstrap into styles.css via CDN
- [x] Update CSS for navigation bar, footer, and responsive design

## Step 3: Update Flask App (app.py)
- [ ] Add routes for /about and /contributors
- [ ] Update predict route to generate chart image with matplotlib
- [ ] Add probability/confidence to prediction
- [ ] Implement PDF download functionality
- [ ] Add navigation context to all routes

## Step 4: Create New Templates
- [ ] Create about.html with ASD info, early detection, methodology
- [ ] Create contributors.html with team members and guide

## Step 5: Update Existing Templates
- [ ] Update index.html to dashboard with title, intro, navigation, footer
- [ ] Update questionnaire.html with navigation and footer
- [ ] Update result.html with navigation, probability, chart, disclaimer, PDF button

## Step 6: Testing and Verification
- [ ] Install new dependencies
- [ ] Run Flask app locally
- [ ] Test all routes and functionality
- [ ] Verify model predictions, chart generation, PDF download
