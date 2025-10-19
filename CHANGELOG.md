# Changelog

All notable changes to the Financial Analysis application will be documented in this file.

## [2.0.0] - 2024-01-XX

### Added
- **Flexible Savings Allocation**: Users can now adjust savings allocation in the investment adjustment interface
  - Savings can be set to any amount at or above the safe savings minimum
  - Remaining funds after allocating to stocks and gold can be directly assigned to savings
  - Real-time validation ensures savings never fall below the safe savings threshold
  
- **Enhanced Dashboard Visualization**:
  - Added interactive pie chart alongside the existing bar chart
  - Pie chart shows percentage distribution with custom labels
  - Custom tooltips display both currency values and percentages
  - Charts displayed side-by-side for better comparison
  
- **Component-Based Dashboard Architecture**:
  - Split dashboard into modular, reusable components
  - `PortfolioOverviewCards`: Summary statistics cards
  - `PortfolioCharts`: Pie and bar chart visualizations
  - `HoldingsTable`: General holdings breakdown
  - `StockBreakdownTable`: Detailed stock holdings view
  - Improved code readability and maintainability
  
- **Updated Adjustment Interface**:
  - New savings slider with minimum constraint visualization
  - Total investment display showing full allocation pool
  - Safe savings minimum clearly indicated
  - Enhanced allocation summary table includes savings
  - Predicted returns now include savings interest rate

### Changed
- **API Response Structure**:
  - `/api/portfolio/[userId]/predictions` now returns:
    - `totalInvestment` instead of `disposableAmount`
    - `safeSavings` minimum amount
    - `currentSavings` current savings allocation
  - `/api/portfolio/[userId]/adjust` now accepts:
    - `proposedSavings` in addition to `proposedAllocations`
    
- **Validation Logic**:
  - Total allocation validation: `savings + gold + stocks = totalInvestment`
  - Savings minimum validation: `proposedSavings >= safeSavings`
  - Updated error messages for clearer user guidance
  
- **Dashboard Layout**:
  - Charts now in 2-column grid for desktop
  - Overview cards remain in 3-column grid
  - Improved responsive behavior for mobile devices
  - Better spacing and visual hierarchy

### Technical Improvements
- Refactored dashboard page from 391 lines to modular component structure
- Separated concerns with dedicated component files
- Improved TypeScript type safety with index signatures
- Enhanced code reusability across dashboard sections
- Better prop interfaces for component communication

### Documentation
- Updated README.md with new features and workflow
- Updated QUICK_START.md with savings adjustment examples
- Enhanced troubleshooting guide
- Added component architecture documentation

## [1.0.0] - 2024-01-XX

### Initial Release
- Two-step onboarding flow
- Location-based user registration
- Stock selection interface
- Portfolio dashboard with bar chart
- Investment adjustment for stocks and gold
- State-specific gold prices
- MongoDB database integration
- Predicted returns calculation
- Real-time portfolio valuation
- Responsive UI with Tailwind CSS
- TypeScript throughout
- Production-ready build system

### Features
- Geolocation API integration
- 40 Indian stocks database
- 28 state gold price tracking
- Savings threshold (percentage/fixed)
- Multi-asset portfolio support
- Zod validation schemas
- External API integration (Polygon, MCX)
- Mock data fallbacks
- Database seeding script

### API Endpoints
- POST /api/onboarding/initial-info
- POST /api/onboarding/select-stocks
- GET /api/assets/stocks
- GET /api/portfolio/[userId]
- GET /api/portfolio/[userId]/predictions
- PUT /api/portfolio/[userId]/adjust
- GET /api/gold-prices/latest

### Database Models
- User with location and preferences
- Asset for stocks and gold
- UserPortfolio for allocations
- GoldPrice for state-wise tracking

### UI Components
- 16 Shadcn/UI components
- Custom AdjustmentDialog
- Responsive layouts
- Interactive charts with Recharts
- Form validation
- Error handling

---

## Version History Summary

- **v2.0.0**: Flexible savings allocation + modular dashboard components
- **v1.0.0**: Initial production release with core features