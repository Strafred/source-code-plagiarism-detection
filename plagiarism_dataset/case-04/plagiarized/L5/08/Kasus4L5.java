public class Kasus4L5 {
    public static void main(String[] args) {
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        for (int i = 1; i <= 10; i++) {
            System.out.println(i + "\t\t" + ConvertMilesToKilo(i));
        }
    }
    private static double ConvertMilesToKilo(int miles) {
        return miles * 1.609;
    }
}
