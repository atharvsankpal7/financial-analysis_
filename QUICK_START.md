# Quick Start Guide

## Prerequisites
- Node.js 18+
- MongoDB (running locally or connection string for remote)
- pnpm package manager

## Installation Steps

### 1. Install Dependencies
```bash
pnpm install
```

### 2. Start MongoDB
Ensure MongoDB is running:
```bash
# For local MongoDB
mongod

# OR verify your MongoDB service is running
```

### 3. Configure Environment
The `.env.local` file is already created with default settings:
```
MONGODB_URI=mongodb://localhost:27017/financial_analysis
```
Update if you're using a different MongoDB instance.

### 4. Seed Database
Populate with 40 Indian stocks and gold prices:
```bash
pnpm seed
```

Expected output:
```
Connecting to MongoDB...
Connected to MongoDB
Seeding stocks...
Seeded 40 stocks
Seeding gold prices...
Seeded gold prices for 28 states
Database seeding completed successfully!
```

### 5. Start Development Server
```bash
pnpm dev
```

### 6. Open Application
Navigate to: http://localhost:3000

You'll be automatically redirected to the onboarding flow.

## Complete User Flow

### Step 1: Initial Information
1. Enter your full name
2. Click "Detect Location" or manually enter state and city
3. Enter initial investment amount (e.g., 100000)
4. Select savings threshold type (Percentage or Fixed Amount)
5. Enter threshold value (e.g., 20 for 20%)
6. Enter annual savings interest rate (e.g., 6.5 for 6.5%)
7. Click "Continue"

### Step 2: Select Stocks
1. Browse the list of 40 Indian stocks
2. Use search bar to filter by name or symbol
3. Select stocks by clicking checkboxes
4. Use "Select All" or "Clear All" for bulk actions
5. Click "Finish Setup" (at least 1 stock required)

### Step 3: Dashboard
You'll see:
- Total portfolio value
- Savings and investment allocations
- Interactive bar chart showing distribution
- Detailed holdings in tabbed view
- Predicted annual returns for each asset

### Step 4: Adjust Investments
1. Click "Adjust Investments" button
2. View your total investment amount and predicted returns
3. Adjust savings allocation (minimum is safe savings amount)
4. Use sliders or input fields to reallocate across stocks and gold
5. Any remaining funds can be allocated to savings
6. Ensure "Remaining to Allocate" shows ₹0.00 (Total must equal initial investment)
7. Click "Confirm Changes"
8. Dashboard refreshes with new allocations

## Sample Test Data

### User Information
- Name: John Doe
- Location: Mumbai, Maharashtra
- Initial Investment: ₹100,000
- Savings Threshold: 20% (₹20,000 safe savings)
- Interest Rate: 6.5%

### Stock Selection Examples
- TCS (Tata Consultancy Services)
- Infosys
- Reliance Industries
- HDFC Bank
- ICICI Bank

### Sample Allocation
You can allocate your total ₹100,000 investment as:
- Savings: ₹25,000 (25% - above the ₹20,000 minimum)
- Gold: ₹30,000 (30%)
- Stocks: ₹45,000 (45%)

Or keep minimum savings:
- Savings: ₹20,000 (20% - minimum safe savings)
- Gold: ₹35,000 (35%)
- Stocks: ₹45,000 (45%)

## Verification Commands

### Check Build
```bash
pnpm build
```

### Check MongoDB Connection
```bash
# Connect to MongoDB shell
mongosh

# Check database
use financial_analysis

# Count documents
db.assets.countDocuments()  // Should return 40
db.goldPrices.countDocuments()  // Should return 28
```

### View Logs
Development server logs appear in terminal showing:
- Compilation status
- API requests
- Database queries

## Troubleshooting

### "MongoDB connection failed"
- Ensure MongoDB is running: `mongod` or check service status
- Verify MONGODB_URI in `.env.local`
- Check MongoDB port (default: 27017)

### "No stocks available"
- Run seed script: `pnpm seed`
- Check terminal for seed success message
- Verify MongoDB connection

### "Build failed"
- Clear cache: `rm -rf .next`
- Reinstall: `rm -rf node_modules && pnpm install`
- Check Node.js version: `node --version` (should be 18+)

### Location detection not working
- Click "Detect Location" and allow browser permission
- Manually enter location if detection fails
- Must use HTTPS in production for geolocation

## Production Build

```bash
# Build for production
pnpm build

# Start production server
pnpm start
```

Production server runs on: http://localhost:3000

## Available Scripts

- `pnpm dev` - Start development server
- `pnpm build` - Build production bundle
- `pnpm start` - Start production server
- `pnpm lint` - Run ESLint
- `pnpm seed` - Seed database with initial data

## API Testing

### Test Stock Listing
```bash
curl http://localhost:3000/api/assets/stocks
```

### Test Gold Prices
```bash
curl "http://localhost:3000/api/gold-prices/latest?state=Maharashtra"
```

## Key Features to Try

1. Complete the onboarding flow with location detection
2. Explore the dashboard with pie and bar charts
3. Try adjusting investments including savings allocation
4. Allocate more than minimum to savings if desired
5. View breakdown by stocks, gold, and savings
6. Check predicted returns for each asset type

## Dashboard Components

The dashboard now uses a modular component structure:
- **PortfolioOverviewCards**: Summary cards at the top
- **PortfolioCharts**: Pie chart and bar chart side-by-side
- **HoldingsTable**: Complete holdings breakdown
- **StockBreakdownTable**: Detailed stock holdings

## Investment Adjustment Features

- Adjust savings (minimum = safe savings threshold)
- Reallocate across all assets (savings, gold, stocks)
- Real-time validation of total allocation
- Flexible savings allocation above minimum
- All changes must total to initial investment amount

For detailed documentation, see `README.md` and `PROJECT_SUMMARY.md`.