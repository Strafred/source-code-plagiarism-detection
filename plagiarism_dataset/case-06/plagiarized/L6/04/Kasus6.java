public class Kasus6 {
    public static void simpanArray(int[] arr, int i) {
        java.util.Scanner sc = new java.util.Scanner(System.in);
        if(i>0) {
            simpanArray(arr, --i);
            System.out.print("Read a number: ");
            arr[i] = sc.nextInt();
        }
    }
    public static void tampilArray(int[] arr, int i) {
        if(i>=0){
            System.out.println(arr[i]);
            tampilArray(arr, --i);
        }
    }
    public static void main(String[] args) {
        int n=10;
        int[] arr = new int[n];
        simpanArray(arr, n);
        tampilArray(arr, --n);
    }
}
