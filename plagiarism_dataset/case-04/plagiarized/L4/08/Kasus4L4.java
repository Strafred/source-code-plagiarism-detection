public class Kasus4L4 {
    public static void main(String[] args) {
        int counter = 1;
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        while (counter <= 10) {
            System.out.println(counter + "\t\t" + ConvertMilesToKilo(counter));
            counter++;
        }
    }
    private static double ConvertMilesToKilo(int miles){
        return miles * 1.609;
    }
}
