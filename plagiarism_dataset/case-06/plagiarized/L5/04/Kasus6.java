public class Kasus6 {
    public static void simpanArray(int[] arr, int n) {
        java.util.Scanner sc = new java.util.Scanner(System.in);
        for (int i = 0; i < n; i++) {
            System.out.print("Read a number: ");
            arr[i] = sc.nextInt();
        }
    }
    public static void tampilArray(int[] arr, int i) {
        System.out.println(arr[i]);
    }
    public static void main(String[] args) {
        int n=10;
        int[] arr = new int[n];
        simpanArray(arr, n);
        for (int i = n-1 ; i >= 0 ; i--) {
            tampilArray(arr, i);
        }
    }
}
