import requests
import sys
import json
from datetime import datetime, date, timedelta

class WasteManagementAPITester:
    def __init__(self, base_url="https://rural-waste-optimize.preview.emergentagent.com"):
        self.base_url = base_url
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.panchayat_id = None
        self.ward_ids = []
        self.vehicle_ids = []
        self.log_ids = []

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.base_url}/api/{endpoint}"
        test_headers = {'Content-Type': 'application/json'}
        if self.token:
            test_headers['Authorization'] = f'Bearer {self.token}'
        if headers:
            test_headers.update(headers)

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=test_headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=test_headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    return True, response.json() if response.text else {}
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test API health"""
        return self.run_test("Health Check", "GET", "", 200)

    def test_signup(self):
        """Test panchayat signup"""
        timestamp = datetime.now().strftime('%H%M%S')
        signup_data = {
            "email": f"test_panchayat_{timestamp}@example.com",
            "name": f"Test Panchayat {timestamp}",
            "address": "Test Village, Test Block, Test District, Test State",
            "password": "TestPass123!"
        }
        
        success, response = self.run_test(
            "Panchayat Signup",
            "POST",
            "auth/signup",
            200,
            data=signup_data
        )
        
        if success and 'token' in response:
            self.token = response['token']
            print(f"   Token obtained: {self.token[:20]}...")
            return True
        return False

    def test_login(self):
        """Test login with existing credentials"""
        # This will fail if signup didn't work, but we'll try anyway
        login_data = {
            "email": "test_panchayat@example.com",
            "password": "TestPass123!"
        }
        
        success, response = self.run_test(
            "Panchayat Login",
            "POST", 
            "auth/login",
            200,
            data=login_data
        )
        
        if success and 'token' in response:
            self.token = response['token']
            return True
        return False

    def test_me_endpoint(self):
        """Test /auth/me endpoint"""
        if not self.token:
            print("❌ Skipping /auth/me test - no token available")
            return False
            
        success, response = self.run_test(
            "Get Current User",
            "GET",
            "auth/me",
            200
        )
        
        if success and ('id' in response or '_id' in response):
            self.panchayat_id = response.get('id') or response.get('_id')
            print(f"   Panchayat ID: {self.panchayat_id}")
            return True
        return False

    def test_create_wards_bulk(self):
        """Test bulk ward creation"""
        if not self.token:
            print("❌ Skipping ward creation - no token")
            return False
            
        wards_data = [
            {"name": "Ward 1", "address": "Ward 1 Area, Test Village"},
            {"name": "Ward 2", "address": "Ward 2 Area, Test Village"},
            {"name": "Ward 3", "address": "Ward 3 Area, Test Village"}
        ]
        
        success, response = self.run_test(
            "Create Wards Bulk",
            "POST",
            "wards/bulk",
            200,
            data=wards_data
        )
        
        if success and isinstance(response, list):
            self.ward_ids = [ward.get('id') or ward.get('_id') for ward in response]
            print(f"   Created {len(self.ward_ids)} wards")
            return True
        return False

    def test_list_wards(self):
        """Test listing wards"""
        if not self.token:
            print("❌ Skipping ward listing - no token")
            return False
            
        success, response = self.run_test(
            "List Wards",
            "GET",
            "wards",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} wards")
            return True
        return False

    def test_create_vehicles_bulk(self):
        """Test bulk vehicle creation"""
        if not self.token:
            print("❌ Skipping vehicle creation - no token")
            return False
            
        vehicles_data = [
            {
                "driver_name": "Driver One",
                "driver_phone": "+919876543210",
                "vehicle_number": "MH12AB1234",
                "capacity": 800
            },
            {
                "driver_name": "Driver Two", 
                "driver_phone": "+919876543211",
                "vehicle_number": "MH12AB1235",
                "capacity": 1000
            }
        ]
        
        success, response = self.run_test(
            "Create Vehicles Bulk",
            "POST",
            "vehicles/bulk",
            200,
            data=vehicles_data
        )
        
        if success and isinstance(response, list):
            self.vehicle_ids = [vehicle.get('id') or vehicle.get('_id') for vehicle in response]
            print(f"   Created {len(self.vehicle_ids)} vehicles")
            return True
        return False

    def test_list_vehicles(self):
        """Test listing vehicles"""
        if not self.token:
            print("❌ Skipping vehicle listing - no token")
            return False
            
        success, response = self.run_test(
            "List Vehicles",
            "GET",
            "vehicles",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} vehicles")
            return True
        return False

    def test_create_log(self):
        """Test creating a waste collection log"""
        if not self.token or not self.ward_ids or not self.vehicle_ids:
            print("❌ Skipping log creation - missing prerequisites")
            return False
            
        log_data = {
            "ward_id": self.ward_ids[0],
            "vehicle_id": self.vehicle_ids[0],
            "waste_collected": 350.5,
            "log_date": date.today().isoformat()
        }
        
        success, response = self.run_test(
            "Create Log",
            "POST",
            "logs",
            200,
            data=log_data
        )
        
        if success and ('id' in response or '_id' in response):
            log_id = response.get('id') or response.get('_id')
            self.log_ids.append(log_id)
            print(f"   Created log with ID: {log_id}")
            return True
        return False

    def test_list_logs(self):
        """Test listing logs"""
        if not self.token:
            print("❌ Skipping log listing - no token")
            return False
            
        success, response = self.run_test(
            "List Logs",
            "GET",
            "logs?limit=50",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} logs")
            return True
        return False

    def test_call_schedule(self):
        """Test call schedule get/set"""
        if not self.token:
            print("❌ Skipping call schedule - no token")
            return False
            
        # Test GET
        success, response = self.run_test(
            "Get Call Schedule",
            "GET",
            "settings/call-schedule",
            200
        )
        
        if not success:
            return False
            
        # Test PUT
        schedule_data = {
            "morning_call_time_ist": "06:30",
            "evening_call_time_ist": "19:30"
        }
        
        success, response = self.run_test(
            "Set Call Schedule",
            "PUT",
            "settings/call-schedule",
            200,
            data=schedule_data
        )
        
        return success

    def test_optimization_run(self):
        """Test running optimization"""
        if not self.token or not self.ward_ids or not self.vehicle_ids:
            print("❌ Skipping optimization - missing prerequisites")
            return False
            
        success, response = self.run_test(
            "Run Optimization",
            "POST",
            "optimization/run",
            200
        )
        
        if success and 'routes_created' in response:
            print(f"   Created {response['routes_created']} routes")
            return True
        return False

    def test_list_routes(self):
        """Test listing routes"""
        if not self.token:
            print("❌ Skipping route listing - no token")
            return False
            
        success, response = self.run_test(
            "List Routes",
            "GET",
            "routes",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} routes")
            return True
        return False

    def test_retell_morning_payload(self):
        """Test Retell morning payload endpoint"""
        if not self.token:
            print("❌ Skipping Retell morning payload - no token")
            return False
            
        # This should return 200 with payload data if routes exist
        success, response = self.run_test(
            "Retell Morning Payload",
            "GET",
            "retell/morning/payload",
            200  # Expecting 200 with data after optimization
        )
        
        return success

    def test_retell_evening_payload(self):
        """Test Retell evening payload endpoint"""
        if not self.token:
            print("❌ Skipping Retell evening payload - no token")
            return False
            
        # This should return 200 with payload data if routes exist
        success, response = self.run_test(
            "Retell Evening Payload",
            "GET",
            "retell/evening/payload", 
            200  # Expecting 200 with data after optimization
        )
        
        return success

    def test_retell_webhook(self):
        """Test Retell webhook endpoint"""
        # This endpoint is unauthenticated but needs existing route data
        webhook_data = {
            "vehicle_number": "MH12AB1234",
            "date": date.today().isoformat(),
            "total_waste_collected": 500.0,
            "wards_visited": ["Ward 1", "Ward 2"],
            "final": True
        }
        
        success, response = self.run_test(
            "Retell Evening Webhook",
            "POST",
            "retell/webhook/evening-report",
            404,  # Expecting 404 if no matching route
            data=webhook_data
        )
        
        return success

    def test_route_allocation_bug_fix(self):
        """Test the route allocation bug fix - ensures ALL vehicles get routes"""
        if not self.token:
            print("❌ Skipping route allocation test - no token")
            return False
            
        print("\n🔧 Testing Route Allocation Bug Fix")
        print("=" * 40)
        
        # First, create a scenario with more vehicles than wards to test edge case
        print("📋 Setting up test scenario...")
        
        # Create 5 vehicles for comprehensive testing
        vehicles_data = [
            {"driver_name": "Driver Alpha", "driver_phone": "+919876543201", "vehicle_number": "TEST001", "capacity": 800},
            {"driver_name": "Driver Beta", "driver_phone": "+919876543202", "vehicle_number": "TEST002", "capacity": 1000},
            {"driver_name": "Driver Gamma", "driver_phone": "+919876543203", "vehicle_number": "TEST003", "capacity": 600},
            {"driver_name": "Driver Delta", "driver_phone": "+919876543204", "vehicle_number": "TEST004", "capacity": 1200},
            {"driver_name": "Driver Echo", "driver_phone": "+919876543205", "vehicle_number": "TEST005", "capacity": 900}
        ]
        
        success, vehicles_response = self.run_test(
            "Create Test Vehicles for Route Allocation",
            "POST",
            "vehicles/bulk",
            200,
            data=vehicles_data
        )
        
        if not success:
            print("❌ Failed to create test vehicles")
            return False
            
        test_vehicle_ids = [v.get('id') or v.get('_id') for v in vehicles_response]
        print(f"✅ Created {len(test_vehicle_ids)} test vehicles")
        
        # Create 3 wards (fewer than vehicles to test edge case)
        wards_data = [
            {"name": "Test Ward Alpha", "address": "Alpha Area, Test Village"},
            {"name": "Test Ward Beta", "address": "Beta Area, Test Village"},
            {"name": "Test Ward Gamma", "address": "Gamma Area, Test Village"}
        ]
        
        success, wards_response = self.run_test(
            "Create Test Wards for Route Allocation",
            "POST",
            "wards/bulk",
            200,
            data=wards_data
        )
        
        if not success:
            print("❌ Failed to create test wards")
            return False
            
        test_ward_ids = [w.get('id') or w.get('_id') for w in wards_response]
        print(f"✅ Created {len(test_ward_ids)} test wards")
        
        # Run optimization
        print("\n🚀 Running optimization...")
        success, opt_response = self.run_test(
            "Run Route Allocation Optimization",
            "POST",
            "optimization/run",
            200
        )
        
        if not success:
            print("❌ Optimization failed")
            return False
            
        routes_created = opt_response.get('routes_created', 0)
        print(f"✅ Optimization created {routes_created} routes")
        
        # Get all routes to analyze allocation
        success, routes_response = self.run_test(
            "Get Routes for Analysis",
            "GET",
            "routes",
            200
        )
        
        if not success:
            print("❌ Failed to get routes")
            return False
            
        routes = routes_response if isinstance(routes_response, list) else []
        print(f"📊 Analyzing {len(routes)} routes...")
        
        # Analysis 1: Check if ALL vehicles have routes
        vehicle_numbers_in_routes = set()
        vehicles_with_wards = 0
        vehicles_without_wards = 0
        total_wards_assigned = 0
        
        for route in routes:
            vehicle_num = route.get('vehicle_number')
            if vehicle_num:
                vehicle_numbers_in_routes.add(vehicle_num)
                wards = route.get('wards', [])
                if wards:
                    vehicles_with_wards += 1
                    total_wards_assigned += len(wards)
                else:
                    vehicles_without_wards += 1
        
        expected_vehicles = set([v['vehicle_number'] for v in vehicles_data])
        
        print(f"\n📈 Route Allocation Analysis:")
        print(f"   Expected vehicles: {len(expected_vehicles)}")
        print(f"   Vehicles with routes: {len(vehicle_numbers_in_routes)}")
        print(f"   Vehicles with wards: {vehicles_with_wards}")
        print(f"   Vehicles without wards: {vehicles_without_wards}")
        print(f"   Total wards assigned: {total_wards_assigned}")
        
        # Test 1: ALL vehicles should have routes (bug fix verification)
        all_vehicles_have_routes = expected_vehicles.issubset(vehicle_numbers_in_routes)
        if all_vehicles_have_routes:
            print("✅ SUCCESS: All vehicles have route assignments")
        else:
            missing_vehicles = expected_vehicles - vehicle_numbers_in_routes
            print(f"❌ FAIL: Missing routes for vehicles: {missing_vehicles}")
            return False
        
        # Test 2: When wards < vehicles, some vehicles should have 0 wards but still have route docs
        if len(test_ward_ids) < len(test_vehicle_ids):
            if vehicles_without_wards > 0:
                print("✅ SUCCESS: Extra vehicles have route docs with 0 wards")
            else:
                print("❌ FAIL: Expected some vehicles to have 0 wards when vehicles > wards")
                return False
        
        # Test 3: Each ward should be assigned to exactly one vehicle
        ward_assignments = {}
        for route in routes:
            for ward in route.get('wards', []):
                ward_id = ward.get('ward_id')
                if ward_id:
                    if ward_id in ward_assignments:
                        print(f"❌ FAIL: Ward {ward_id} assigned to multiple vehicles")
                        return False
                    ward_assignments[ward_id] = route.get('vehicle_number')
        
        print(f"✅ SUCCESS: {len(ward_assignments)} wards uniquely assigned")
        
        # Test 4: Verify Retell payloads include ALL drivers
        print("\n📞 Testing Retell payload completeness...")
        
        success, morning_payload = self.run_test(
            "Retell Morning Payload - All Drivers",
            "GET",
            "retell/morning/payload",
            200
        )
        
        if success:
            drivers_in_morning = morning_payload.get('drivers', [])
            morning_vehicle_numbers = set([d.get('vehicle_number') for d in drivers_in_morning])
            
            if expected_vehicles.issubset(morning_vehicle_numbers):
                print("✅ SUCCESS: Morning payload includes all drivers")
            else:
                missing_morning = expected_vehicles - morning_vehicle_numbers
                print(f"❌ FAIL: Morning payload missing drivers: {missing_morning}")
                return False
        else:
            print("❌ FAIL: Could not get morning payload")
            return False
        
        success, evening_payload = self.run_test(
            "Retell Evening Payload - All Drivers",
            "GET",
            "retell/evening/payload",
            200
        )
        
        if success:
            drivers_in_evening = evening_payload.get('drivers', [])
            evening_vehicle_numbers = set([d.get('vehicle_number') for d in drivers_in_evening])
            
            if expected_vehicles.issubset(evening_vehicle_numbers):
                print("✅ SUCCESS: Evening payload includes all drivers")
            else:
                missing_evening = expected_vehicles - evening_vehicle_numbers
                print(f"❌ FAIL: Evening payload missing drivers: {missing_evening}")
                return False
        else:
            print("❌ FAIL: Could not get evening payload")
            return False
        
        # Cleanup test data
        print("\n🧹 Cleaning up test data...")
        for vehicle_id in test_vehicle_ids:
            self.run_test(f"Delete Test Vehicle {vehicle_id}", "DELETE", f"vehicles/{vehicle_id}", 200)
        
        for ward_id in test_ward_ids:
            self.run_test(f"Delete Test Ward {ward_id}", "DELETE", f"wards/{ward_id}", 200)
        
        print("✅ Route allocation bug fix verification PASSED")
        return True

    def test_route_allocation_equal_scenario(self):
        """Test route allocation when wards == vehicles"""
        if not self.token:
            print("❌ Skipping equal scenario test - no token")
            return False
            
        print("\n⚖️ Testing Equal Wards/Vehicles Scenario")
        print("=" * 40)
        
        # Create 3 vehicles and 3 wards
        vehicles_data = [
            {"driver_name": "Equal Driver 1", "driver_phone": "+919876543301", "vehicle_number": "EQ001", "capacity": 800},
            {"driver_name": "Equal Driver 2", "driver_phone": "+919876543302", "vehicle_number": "EQ002", "capacity": 900},
            {"driver_name": "Equal Driver 3", "driver_phone": "+919876543303", "vehicle_number": "EQ003", "capacity": 1000}
        ]
        
        wards_data = [
            {"name": "Equal Ward 1", "address": "Equal Area 1, Test Village"},
            {"name": "Equal Ward 2", "address": "Equal Area 2, Test Village"},
            {"name": "Equal Ward 3", "address": "Equal Area 3, Test Village"}
        ]
        
        # Create vehicles
        success, vehicles_response = self.run_test(
            "Create Equal Test Vehicles",
            "POST",
            "vehicles/bulk",
            200,
            data=vehicles_data
        )
        
        if not success:
            return False
            
        # Create wards
        success, wards_response = self.run_test(
            "Create Equal Test Wards",
            "POST",
            "wards/bulk",
            200,
            data=wards_data
        )
        
        if not success:
            return False
            
        # Run optimization
        success, opt_response = self.run_test(
            "Run Equal Scenario Optimization",
            "POST",
            "optimization/run",
            200
        )
        
        if not success:
            return False
            
        # Get routes
        success, routes_response = self.run_test(
            "Get Equal Scenario Routes",
            "GET",
            "routes",
            200
        )
        
        if not success:
            return False
            
        routes = routes_response if isinstance(routes_response, list) else []
        
        # Verify each vehicle has at least 1 ward
        vehicles_with_wards = 0
        for route in routes:
            vehicle_num = route.get('vehicle_number')
            if vehicle_num and vehicle_num.startswith('EQ'):
                wards = route.get('wards', [])
                if wards:
                    vehicles_with_wards += 1
                    print(f"   Vehicle {vehicle_num}: {len(wards)} wards")
        
        if vehicles_with_wards == 3:
            print("✅ SUCCESS: All vehicles have at least 1 ward in equal scenario")
            result = True
        else:
            print(f"❌ FAIL: Only {vehicles_with_wards}/3 vehicles have wards")
            result = False
        
        # Cleanup
        for v in vehicles_response:
            vehicle_id = v.get('id') or v.get('_id')
            self.run_test(f"Delete Equal Vehicle {vehicle_id}", "DELETE", f"vehicles/{vehicle_id}", 200)
        
        for w in wards_response:
            ward_id = w.get('id') or w.get('_id')
            self.run_test(f"Delete Equal Ward {ward_id}", "DELETE", f"wards/{ward_id}", 200)
        
        return result

    def test_ward_crud(self):
        """Test individual ward CRUD operations"""
        if not self.token:
            print("❌ Skipping ward CRUD - no token")
            return False
            
        # Create single ward
        ward_data = {"name": "Test Ward CRUD", "address": "CRUD Test Address"}
        success, response = self.run_test(
            "Create Single Ward",
            "POST",
            "wards",
            200,
            data=ward_data
        )
        
        if not success or ('id' not in response and '_id' not in response):
            return False
            
        ward_id = response.get('id') or response.get('_id')
        
        # Update ward
        update_data = {"name": "Updated Ward CRUD", "address": "Updated CRUD Address"}
        success, response = self.run_test(
            "Update Ward",
            "PUT",
            f"wards/{ward_id}",
            200,
            data=update_data
        )
        
        if not success:
            return False
            
        # Delete ward
        success, response = self.run_test(
            "Delete Ward",
            "DELETE",
            f"wards/{ward_id}",
            200
        )
        
        return success

    def test_vehicle_crud(self):
        """Test individual vehicle CRUD operations"""
        if not self.token:
            print("❌ Skipping vehicle CRUD - no token")
            return False
            
        # Create single vehicle
        vehicle_data = {
            "driver_name": "CRUD Driver",
            "driver_phone": "+919999999999",
            "vehicle_number": "TEST1234",
            "capacity": 500
        }
        success, response = self.run_test(
            "Create Single Vehicle",
            "POST",
            "vehicles",
            200,
            data=vehicle_data
        )
        
        if not success or ('id' not in response and '_id' not in response):
            return False
            
        vehicle_id = response.get('id') or response.get('_id')
        
        # Update vehicle
        update_data = {
            "driver_name": "Updated CRUD Driver",
            "driver_phone": "+919999999998",
            "vehicle_number": "UPDATED1234",
            "capacity": 600
        }
        success, response = self.run_test(
            "Update Vehicle",
            "PUT",
            f"vehicles/{vehicle_id}",
            200,
            data=update_data
        )
        
        if not success:
            return False
            
        # Delete vehicle
        success, response = self.run_test(
            "Delete Vehicle",
            "DELETE",
            f"vehicles/{vehicle_id}",
            200
        )
        
        return success

def main():
    print("🚀 Starting Waste Management API Tests")
    print("=" * 50)
    
    tester = WasteManagementAPITester()
    
    # Test sequence
    tests = [
        ("Health Check", tester.test_health_check),
        ("Signup", tester.test_signup),
        ("Auth Me", tester.test_me_endpoint),
        ("Create Wards Bulk", tester.test_create_wards_bulk),
        ("List Wards", tester.test_list_wards),
        ("Create Vehicles Bulk", tester.test_create_vehicles_bulk),
        ("List Vehicles", tester.test_list_vehicles),
        ("Create Log", tester.test_create_log),
        ("List Logs", tester.test_list_logs),
        ("Call Schedule", tester.test_call_schedule),
        ("Run Optimization", tester.test_optimization_run),
        ("List Routes", tester.test_list_routes),
        ("Retell Morning Payload", tester.test_retell_morning_payload),
        ("Retell Evening Payload", tester.test_retell_evening_payload),
        ("Retell Webhook", tester.test_retell_webhook),
        ("Ward CRUD", tester.test_ward_crud),
        ("Vehicle CRUD", tester.test_vehicle_crud),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} - Exception: {str(e)}")
            failed_tests.append(test_name)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if failed_tests:
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        return 1
    else:
        print("✅ All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())