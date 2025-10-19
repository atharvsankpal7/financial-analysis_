# Separated Authentication & Onboarding Architecture

## Overview

This document describes the refactored architecture that separates **login data** from **onboarding data** in the Financial Analysis application.

---

## Architecture Design

### Previous Architecture (Single Model)
```
User Model:
├── email (login data)
├── password (login data)
├── fullName (onboarding data)
├── location (onboarding data)
├── initialInvestmentAmount (onboarding data)
├── savingsThreshold (onboarding data)
└── annualSavingsInterestRate (onboarding data)
```

**Problems:**
- Mixed concerns (authentication + business logic)
- Complex optional field handling
- Difficult to maintain and extend
- Null checks required throughout the codebase

### New Architecture (Separated Models)

```
┌─────────────────────────────────────────────────────────────┐
│                    User Model                                │
│  (Authentication Data Only)                                  │
├─────────────────────────────────────────────────────────────┤
│  • email: string (required)                                  │
│  • password: string (required, select: false)                │
│  • createdAt: Date                                           │
│  • updatedAt: Date                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:1 relationship
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OnboardingData Model                            │
│  (Profile & Investment Data)                                 │
├─────────────────────────────────────────────────────────────┤
│  • userId: ObjectId (reference to User)                      │
│  • fullName: string (required)                               │
│  • location: object (required)                               │
│     - state: string                                          │
│     - city: string                                           │
│     - country: string                                        │
│     - coordinates: { lat, lng } (optional)                   │
│  • initialInvestmentAmount: number (required)                │
│  • savingsThreshold: object (required)                       │
│     - type: "percentage" | "fixed"                           │
│     - value: number                                          │
│  • annualSavingsInterestRate: number (required)              │
│  • createdAt: Date                                           │
│  • updatedAt: Date                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:1 relationship
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              UserPortfolio Model                             │
│  (Investment Portfolio Data)                                 │
├─────────────────────────────────────────────────────────────┤
│  • userId: ObjectId (reference to User)                      │
│  • selectedStockIds: string[] (stock UUIDs)                  │
│  • allocations: Record<string, number>                       │
│  • goldAllocation: number                                    │
│  • savingsAllocation: number                                 │
│  • onboardingComplete: boolean                               │
│  • createdAt: Date                                           │
│  • updatedAt: Date                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Benefits of Separation

### 1. **Clear Separation of Concerns**
- Authentication logic isolated from business logic
- Easier to understand and maintain
- Better adheres to Single Responsibility Principle

### 2. **Type Safety**
- No optional fields in models (except where truly optional)
- No null checks scattered throughout the code
- TypeScript can enforce data completeness

### 3. **Flexible Onboarding**
- Can add/remove onboarding steps without affecting auth
- Easy to implement multi-step onboarding flows
- Can update profile data independently

### 4. **Better Security**
- Login credentials completely separate from business data
- Easier to implement different access controls
- Clearer audit trails

### 5. **Scalability**
- Can add more onboarding models (e.g., KYC, preferences)
- Easy to implement progressive data collection
- Better database indexing strategies

---

## Data Flow

### 1. Sign Up Flow
```
User submits email + password
         ↓
POST /api/auth/signup
         ↓
Create User document
  • email: provided
  • password: hashed
         ↓
Create empty UserPortfolio
  • userId: new user ID
  • onboardingComplete: false
         ↓
Return: { userId, email }
         ↓
Auto sign-in user
```

### 2. Sign In Flow
```
User submits credentials
         ↓
POST /api/auth/signin (NextAuth)
         ↓
Verify email + password
         ↓
Create JWT session
         ↓
Check onboarding status
         ↓
Redirect based on status:
  • No profile data? → /onboarding/initial-info
  • No stocks? → /onboarding/select-stocks
  • Complete? → /dashboard/[userId]
```

### 3. Onboarding Step 1: Profile Info
```
User submits profile data
         ↓
POST /api/onboarding/initial-info
  (authenticated via session)
         ↓
Create/Update OnboardingData document
  • userId: from session
  • fullName, location, investment details
         ↓
Ensure UserPortfolio exists
         ↓
Return: { userId, safeSavings }
         ↓
Navigate to stock selection
```

### 4. Onboarding Step 2: Stock Selection
```
User selects stocks
         ↓
POST /api/onboarding/select-stocks
  (authenticated via session)
         ↓
Verify OnboardingData exists
         ↓
Update UserPortfolio
  • selectedStockIds: provided
  • allocations: initialized to 0
  • onboardingComplete: true
         ↓
Return: { portfolioId, userId }
         ↓
Navigate to dashboard
```

### 5. Status Check
```
App loads or route change
         ↓
GET /api/auth/status
  (authenticated via session)
         ↓
Check:
  • User exists?
  • OnboardingData exists? (hasProfileInfo)
  • Portfolio has stocks? (hasStockSelection)
  • Portfolio.onboardingComplete? (hasCompletedOnboarding)
         ↓
Return status object
         ↓
Frontend routes accordingly
```

---

## API Endpoints

### Authentication Endpoints

#### POST /api/auth/signup
**Purpose:** Create new user account (login credentials only)

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Account created successfully",
  "data": {
    "userId": "507f1f77bcf86cd799439011",
    "email": "user@example.com"
  }
}
```

**What it creates:**
- User document (email + hashed password)
- Empty UserPortfolio document

---

#### POST /api/auth/signin
**Purpose:** Authenticate user (via NextAuth)

**Handled by:** NextAuth Credentials Provider

**What it does:**
- Verifies credentials
- Creates JWT session
- Sets HTTP-only cookies

---

#### GET /api/auth/status
**Purpose:** Check authentication and onboarding status

**Authentication:** Required (session)

**Response:**
```json
{
  "success": true,
  "data": {
    "isAuthenticated": true,
    "hasCompletedOnboarding": true,
    "hasProfileInfo": true,
    "hasStockSelection": true,
    "userId": "507f1f77bcf86cd799439011",
    "email": "user@example.com",
    "fullName": "John Doe"
  }
}
```

**Logic:**
- `hasProfileInfo`: OnboardingData document exists
- `hasStockSelection`: Portfolio has selectedStockIds.length > 0
- `hasCompletedOnboarding`: hasProfileInfo AND hasStockSelection AND portfolio.onboardingComplete === true

---

### Onboarding Endpoints

#### POST /api/onboarding/initial-info
**Purpose:** Save user profile and investment information

**Authentication:** Required (session)

**Request:**
```json
{
  "fullName": "John Doe",
  "location": {
    "state": "Maharashtra",
    "city": "Mumbai",
    "country": "India",
    "coordinates": { "lat": 19.076, "lng": 72.8777 }
  },
  "initialInvestmentAmount": 100000,
  "savingsThreshold": {
    "type": "percentage",
    "value": 20
  },
  "annualSavingsInterestRate": 6.5
}
```

**What it does:**
1. Gets userId from session
2. Creates or updates OnboardingData document
3. Ensures UserPortfolio exists
4. Calculates safe savings
5. Returns success with calculated values

**Note:** Does NOT mark onboarding as complete

---

#### POST /api/onboarding/select-stocks
**Purpose:** Save stock selections and complete onboarding

**Authentication:** Required (session)

**Request:**
```json
{
  "selectedStockIds": [
    "550e8400-e29b-41d4-a716-446655440001",
    "550e8400-e29b-41d4-a716-446655440002",
    "550e8400-e29b-41d4-a716-446655440003"
  ]
}
```

**What it does:**
1. Gets userId from session
2. Verifies OnboardingData exists (profile completed first)
3. Checks onboarding not already complete
4. Initializes allocations for stocks (all 0)
5. Updates UserPortfolio with stocks
6. Sets onboardingComplete = true
7. Returns portfolio data

**Note:** This marks onboarding as complete

---

### Profile & Portfolio Endpoints

All these endpoints now fetch OnboardingData separately from User:

#### GET /api/profile/[userId]
- Fetches User + OnboardingData
- Returns combined profile information

#### PUT /api/profile/[userId]
- Updates OnboardingData document
- Validates investment changes against portfolio
- Returns updated profile

#### GET /api/portfolio/[userId]
- Fetches User + OnboardingData + UserPortfolio
- Uses OnboardingData for investment amounts, location, etc.
- Returns complete portfolio view

#### GET /api/portfolio/[userId]/predictions
- Uses OnboardingData for calculations
- Returns predicted returns

#### PUT /api/portfolio/[userId]/adjust
- Uses OnboardingData for validation
- Updates UserPortfolio allocations

---

## Database Schema

### User Collection
```javascript
{
  _id: ObjectId("507f1f77bcf86cd799439011"),
  email: "user@example.com",
  password: "$2a$10$hashed_password_here",
  createdAt: ISODate("2024-01-15T10:30:00Z"),
  updatedAt: ISODate("2024-01-15T10:30:00Z")
}
```

**Indexes:**
- email (unique)

---

### OnboardingData Collection
```javascript
{
  _id: ObjectId("507f1f77bcf86cd799439012"),
  userId: ObjectId("507f1f77bcf86cd799439011"),
  fullName: "John Doe",
  location: {
    state: "Maharashtra",
    city: "Mumbai",
    country: "India",
    coordinates: {
      lat: 19.076,
      lng: 72.8777
    }
  },
  initialInvestmentAmount: 100000,
  savingsThreshold: {
    type: "percentage",
    value: 20
  },
  annualSavingsInterestRate: 6.5,
  createdAt: ISODate("2024-01-15T10:35:00Z"),
  updatedAt: ISODate("2024-01-15T10:35:00Z")
}
```

**Indexes:**
- userId (unique, indexed)

**Relationships:**
- References User._id

---

### UserPortfolio Collection
```javascript
{
  _id: ObjectId("507f1f77bcf86cd799439013"),
  userId: ObjectId("507f1f77bcf86cd799439011"),
  selectedStockIds: [
    "550e8400-e29b-41d4-a716-446655440001",
    "550e8400-e29b-41d4-a716-446655440002"
  ],
  allocations: {
    "550e8400-e29b-41d4-a716-446655440001": 40000,
    "550e8400-e29b-41d4-a716-446655440002": 30000
  },
  goldAllocation: 20000,
  savingsAllocation: 10000,
  onboardingComplete: true,
  createdAt: ISODate("2024-01-15T10:40:00Z"),
  updatedAt: ISODate("2024-01-15T11:20:00Z")
}
```

**Indexes:**
- userId (unique, indexed)

**Relationships:**
- References User._id

---

## Code Organization

### Models
```
src/models/
├── User.ts                    # Authentication data only
├── OnboardingData.ts          # Profile & investment data
├── UserPortfolio.ts           # Portfolio & allocations
├── Asset.ts                   # Stock data
└── GoldPrice.ts               # Gold price data
```

### API Routes
```
src/app/api/
├── auth/
│   ├── [...nextauth]/route.ts    # NextAuth handler
│   ├── signup/route.ts            # Create account
│   └── status/route.ts            # Check onboarding status
├── onboarding/
│   ├── initial-info/route.ts      # Step 1: Profile
│   └── select-stocks/route.ts     # Step 2: Stocks
├── profile/[userId]/route.ts      # Get/Update profile
└── portfolio/[userId]/
    ├── route.ts                   # Get portfolio
    ├── predictions/route.ts       # Get predictions
    └── adjust/route.ts            # Adjust allocations
```

---

## Frontend Integration

### Minimal Frontend Changes Required

#### 1. Signup Page
**Remove:** Onboarding fields from signup form
**Keep:** Only email + password

```typescript
// Before: sending extra fields
const response = await fetch('/api/auth/signup', {
  body: JSON.stringify({ email, password, fullName: '', ... })
});

// After: only login credentials
const response = await fetch('/api/auth/signup', {
  body: JSON.stringify({ email, password })
});
```

---

#### 2. Signin Page
**Add:** Status check after login

```typescript
const result = await signIn('credentials', { email, password, redirect: false });

if (result?.ok) {
  const statusRes = await fetch('/api/auth/status');
  const { data } = await statusRes.json();
  
  // Route based on status
  if (!data.hasProfileInfo) {
    router.push('/onboarding/initial-info');
  } else if (!data.hasStockSelection) {
    router.push('/onboarding/select-stocks');
  } else {
    router.push(`/dashboard/${data.userId}`);
  }
}
```

---

#### 3. Onboarding Pages
**Remove:** userId from request body (uses session)

```typescript
// Before
await fetch('/api/onboarding/initial-info', {
  body: JSON.stringify({ userId, fullName, ... })
});

// After
await fetch('/api/onboarding/initial-info', {
  body: JSON.stringify({ fullName, ... }) // No userId
});
```

---

### Optional: AuthGuard Component

```typescript
export function AuthGuard({ children }) {
  const { data: session, status } = useSession();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    async function checkStatus() {
      if (status === 'unauthenticated') {
        router.push('/auth/signin');
        return;
      }

      const res = await fetch('/api/auth/status');
      const { data } = await res.json();

      if (!data.hasCompletedOnboarding) {
        if (!data.hasProfileInfo) {
          router.push('/onboarding/initial-info');
        } else if (!data.hasStockSelection) {
          router.push('/onboarding/select-stocks');
        }
      }

      setIsChecking(false);
    }

    checkStatus();
  }, [status]);

  if (isChecking) return <LoadingSpinner />;

  return <>{children}</>;
}
```

---

## Migration Guide

### For Existing Data

If you have existing users with all data in the User model:

1. **Create migration script:**
```javascript
// scripts/migrate-to-separated-models.js
async function migrate() {
  const users = await User.find({});
  
  for (const user of users) {
    // Only migrate if user has onboarding data
    if (user.fullName && user.initialInvestmentAmount) {
      await OnboardingData.create({
        userId: user._id,
        fullName: user.fullName,
        location: user.location,
        initialInvestmentAmount: user.initialInvestmentAmount,
        savingsThreshold: user.savingsThreshold,
        annualSavingsInterestRate: user.annualSavingsInterestRate,
      });
      
      console.log(`Migrated user: ${user.email}`);
    }
  }
}
```

2. **Run migration:**
```bash
node scripts/migrate-to-separated-models.js
```

3. **Verify:**
```javascript
// Check counts match
const userCount = await User.countDocuments({ fullName: { $exists: true } });
const onboardingCount = await OnboardingData.countDocuments({});
console.log(`Users with data: ${userCount}, Migrated: ${onboardingCount}`);
```

---

## Testing Strategy

### Unit Tests

#### 1. Model Tests
```javascript
describe('OnboardingData Model', () => {
  it('should require userId', async () => {
    const data = new OnboardingData({ fullName: 'Test' });
    await expect(data.save()).rejects.toThrow();
  });

  it('should create valid onboarding data', async () => {
    const data = await OnboardingData.create({
      userId: new mongoose.Types.ObjectId(),
      fullName: 'Test User',
      location: { state: 'MH', city: 'Mumbai', country: 'India' },
      initialInvestmentAmount: 100000,
      savingsThreshold: { type: 'percentage', value: 20 },
      annualSavingsInterestRate: 6.5,
    });
    expect(data).toBeDefined();
  });
});
```

#### 2. API Tests
```javascript
describe('POST /api/auth/signup', () => {
  it('should create user with only email and password', async () => {
    const res = await request(app)
      .post('/api/auth/signup')
      .send({ email: 'test@example.com', password: 'test123' });
    
    expect(res.status).toBe(201);
    expect(res.body.data.userId).toBeDefined();
    
    // Verify OnboardingData doesn't exist yet
    const onboarding = await OnboardingData.findOne({ userId: res.body.data.userId });
    expect(onboarding).toBeNull();
  });
});

describe('POST /api/onboarding/initial-info', () => {
  it('should create onboarding data for authenticated user', async () => {
    // Assumes session setup
    const res = await request(app)
      .post('/api/onboarding/initial-info')
      .set('Cookie', sessionCookie)
      .send(profileData);
    
    expect(res.status).toBe(200);
    expect(res.body.data.userId).toBeDefined();
  });
});
```

### Integration Tests

```javascript
describe('Complete Onboarding Flow', () => {
  it('should complete full signup and onboarding', async () => {
    // 1. Signup
    const signupRes = await request(app)
      .post('/api/auth/signup')
      .send({ email: 'new@example.com', password: 'pass123' });
    expect(signupRes.status).toBe(201);
    
    // 2. Signin
    const signinRes = await signIn('new@example.com', 'pass123');
    const session = signinRes.sessionCookie;
    
    // 3. Check status (should be incomplete)
    const statusRes1 = await request(app)
      .get('/api/auth/status')
      .set('Cookie', session);
    expect(statusRes1.body.data.hasCompletedOnboarding).toBe(false);
    
    // 4. Submit profile info
    const profileRes = await request(app)
      .post('/api/onboarding/initial-info')
      .set('Cookie', session)
      .send(profileData);
    expect(profileRes.status).toBe(200);
    
    // 5. Select stocks
    const stocksRes = await request(app)
      .post('/api/onboarding/select-stocks')
      .set('Cookie', session)
      .send({ selectedStockIds: ['stock1', 'stock2'] });
    expect(stocksRes.status).toBe(200);
    
    // 6. Check status (should be complete)
    const statusRes2 = await request(app)
      .get('/api/auth/status')
      .set('Cookie', session);
    expect(statusRes2.body.data.hasCompletedOnboarding).toBe(true);
  });
});
```

---

## Security Considerations

### 1. Authentication
- ✅ Passwords hashed with bcrypt (10 rounds)
- ✅ Password field has `select: false` in schema
- ✅ JWT tokens for session management
- ✅ HTTP-only cookies prevent XSS

### 2. Authorization
- ✅ All onboarding endpoints require valid session
- ✅ userId taken from session, not request body
- ✅ Users can only access their own data

### 3. Data Validation
- ✅ Zod schemas for all inputs
- ✅ Mongoose schemas with required fields
- ✅ Type safety with TypeScript

### 4. Database Security
- ✅ Unique indexes on email and userId
- ✅ Foreign key constraints via ObjectId references
- ✅ Proper error handling (no sensitive data leaked)

---

## Performance Considerations

### 1. Database Queries
- Most routes now require 2-3 queries (User + OnboardingData + Portfolio)
- Consider using MongoDB aggregation for complex queries
- Add indexes on frequently queried fields

### 2. Optimization Strategies

#### Caching
```javascript
// Cache onboarding data in session
callbacks: {
  async session({ session, token }) {
    const onboarding = await OnboardingData.findOne({ userId: token.id });
    session.user.hasCompletedOnboarding = !!onboarding;
    return session;
  }
}
```

#### Aggregation
```javascript
// Fetch all user data in one query
const userData = await User.aggregate([
  { $match: { _id: userId } },
  {
    $lookup: {
      from: 'onboardingdatas',
      localField: '_id',
      foreignField: 'userId',
      as: 'onboarding'
    }
  },
  {
    $lookup: {
      from: 'userportfolios',
      localField: '_id',
      foreignField: 'userId',
      as: 'portfolio'
    }
  }
]);
```

#### Lean Queries
```javascript
// Use .lean() for read-only operations
const onboarding = await OnboardingData.findOne({ userId }).lean();
```

---

## Troubleshooting

### Issue: "Onboarding data not found"
**Cause:** User completed signup but hasn't filled profile info
**Solution:** Check `/api/auth/status` and redirect to appropriate step

### Issue: "Unauthorized" on onboarding endpoints
**Cause:** Session expired or invalid
**Solution:** Re-authenticate user

### Issue: Can't save onboarding data
**Cause:** Validation errors or missing required fields
**Solution:** Check error response for specific field issues

### Issue: User stuck in onboarding loop
**Cause:** `onboardingComplete` flag not set
**Solution:** Verify stock selection endpoint sets flag correctly

---

## Future Enhancements

### 1. Additional Onboarding Steps
```javascript
// Example: KYC verification
const KYCData = new Schema({
  userId: ObjectId,
  panCard: String,
  aadharCard: String,
  verified: Boolean,
});
```

### 2. Progressive Data Collection
- Collect minimal data initially
- Request additional info over time
- Use analytics to optimize timing

### 3. Multi-language Support
- Store language preference in OnboardingData
- Localize onboarding flow

### 4. Social Login
- Add OAuth providers (Google, Facebook)
- Create User with provider info
- Still require onboarding data

---

## Summary

The separated architecture provides:

✅ **Clear separation** between authentication and business data  
✅ **Type safety** with no optional fields in core models  
✅ **Flexibility** for multi-step onboarding flows  
✅ **Security** with proper data isolation  
✅ **Scalability** for future feature additions  
✅ **Maintainability** with cleaner code organization  

**Key Files:**
- `src/models/User.ts` - Login credentials only
- `src/models/OnboardingData.ts` - Profile & investment data
- `src/app/api/auth/status/route.ts` - Check onboarding progress
- `src/app/api/onboarding/initial-info/route.ts` - Save profile
- `src/app/api/onboarding/select-stocks/route.ts` - Complete onboarding

**Frontend Changes Required:**
- Minimal (remove userId from requests, add status checks)
- Documented in `FRONTEND_MIGRATION_GUIDE.md`

**Backend is now fully implemented and ready for frontend integration!**