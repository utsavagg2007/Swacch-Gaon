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
            
        # This should return 404 if no routes exist
        success, response = self.run_test(
            "Retell Morning Payload",
            "GET",
            "retell/morning/payload",
            404  # Expecting 404 initially
        )
        
        return success

    def test_retell_evening_payload(self):
        """Test Retell evening payload endpoint"""
        if not self.token:
            print("❌ Skipping Retell evening payload - no token")
            return False
            
        # This should return 404 if no routes exist
        success, response = self.run_test(
            "Retell Evening Payload",
            "GET",
            "retell/evening/payload", 
            404  # Expecting 404 initially
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
        
        if not success or 'id' not in response:
            return False
            
        vehicle_id = response['id']
        
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